"""
ClassifierLSTM module for Greenland supraglacial lake drainage classification.

Combines multiple components to classify supraglacial lake drainage events
from satellite imagery sequences and scalar time series data.
"""
import torch
import torch.nn as nn

from .blocks import FrontCNN, ScalarLSTM, ClassHeadMLP, GlobalPooling
from .clstm import CLSTM
from .attention import SpatialCBAM, FullCBAM


# Channel name to index mapping for NC files
CHANNEL_NAMES = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'mask']

class LakeDrainageClassifier(nn.Module):
    """
    Lake drainage classification model.
    
    Processes multi-modal inputs (satellite imagery sequences, water area sequences, cloud coverage)
    to classify Greenland supraglacial lake drainage events into four categories:
    ND (no drainage), ED (englacial drainage), LD (lateral drainage), CD (crevasse drainage).

    The model supports flexible configurations:
    - image sequences with optional spatial attention mechanisms
    - scalar time series (water area, cloud coverage)
    - various attention options (e.g., none, spatial CBAM, full CBAM, architectural)

    Args:
        use_imgseq              (bool): whether to use image sequence processing (default: True)
        use_areaseq             (bool): whether to use water area time series (default: True)
        use_cloudyseq           (bool): whether to use cloud coverage time series (default: False)
        learn_area_weights      (bool): whether to learn per-timestep weights for area_seq (default: False)
                                        When True, learns a [seq_len] vector of logits that are sigmoided
                                        and multiplied with area_seq before processing.
        learn_cloudy_weights    (bool): whether to learn per-timestep weights for cloudy_seq (default: False)
                                        When True, learns a [seq_len] vector of logits that are sigmoided
                                        and multiplied with cloudy_seq before processing.
        seq_len                 (int): sequence length, required when learn_*_weights=True (default: 153)
        use_nir                 (bool): whether to include NIR band in imagery (default: False)
        use_swir16              (bool): whether to include SWIR16 band in imagery (default: False)
        use_swir22              (bool): whether to include SWIR22 band in imagery (default: False)
        attention_type          (str): type of attention mechanism. Options:
            - 'none': no attention
            - 'spatial': spatial CBAM
            - 'full': full CBAM (channel + spatial)
            - 'arch': architectural attention, dual pathway with lake mask
            default: 'none'
        num_classes             (int): number of output classes (default: 4)
        input_H                 (int): input image height (default: 512)
        input_W                 (int): input image width (default: 512)
        frontcnn_base_channels  (int) base channels for FrontCNN (default: 8)
        frontcnn_num_layers     (int): number of layers in frontcnn (default: 4)
        frontcnn_out_hw         (tuple): output spatial dimensions after FrontCNN (default: (64,64))
        frontcnn_pool           (str): pooling type for FrontCNN ('max, 'avg', 'none') (default: 'max')
        clstm_hidden            (int): hidden channels for CLSTM (default: 32)
        clstm_kernel            (int): kernel size for CLSTM (default: 3)
        slstm_hidden            (int): hidden dimension for scalar LSTMs (default: 16)
        slstm_num_layers        (int): number of layers for scalar LSTMs (default: 1)
        slstm_dropout           (float): dropout for scalar LSTMs (default: 0.0)
        classhead_hidden        (int): hidden dimension for classification head MLP (default: 64)
        classhead_dropout       (float): dropout for classification head MLP (default: 0.0)
        attention_reduction     (int): channel reduction ratio for full CBAM (default: 16)
        attention_kernel        (int): kernel size for spatial attention (default: 7)
        pool_type               (str): pooling type before classification head ('max', 'avg') (default: 'avg')

    Input:
        x: [B, T, C, H, W] tensor of image sequences
            B: batch size
            T: time steps
            C: input image channels (RGB + optional NIR/SWIR + mask)
               Base: 3 RGB + 1 mask = 4 channels
               With use_nir=True: +1 channel
               With use_swir16=True: +1 channel
               With use_swir22=True: +1 channel
               Max: 7 channels (RGB + NIR + SWIR16 + SWIR22 + mask)
            H: height
            W: width
        area_seq: [B, T, 1] tensor of water area time series (optional)
        cloudy_seq: [B, T, 1] tensor of cloud coverage time series (optional)

    Output:
        [B, num_classes] tensor of class logits

    Example:
        >>> model = LakeDrainageClassifier(
                use_imgseq=True,
                use_areaseq=True,
                use_cloudyseq=False,
                use_nir=True,
                use_swir16=True,
                attention_type='spatial',
            )
        >>> x = torch.randn(16, 153, 6, 512, 512)          # image sequences [B=16, T=153, C=6, H=512, W=512]
        >>> area_seq = torch.randn(16, 153, 1)              # water area sequences [B=16, T=153, 1]
        >>> logits = model(x, area_seq)                    # output logits [B=16, num_classes=4]
    """
    def __init__(
        self,
        # feature flags
        use_imgseq=True,
        use_areaseq=True,
        use_cloudyseq=False,
        learn_area_weights=False,
        learn_cloudy_weights=False,
        seq_len=153,
        # spectral band flags (beyond RGB)
        use_nir=False,
        use_swir16=False,
        use_swir22=False,
        # attention configuration
        attention_type='none',
        # classification and imagery configuration (these may not change)
        num_classes=4,
        input_H=512,
        input_W=512,
        # FrontCNN configuration
        frontcnn_base_channels=8,
        frontcnn_num_layers=4,
        frontcnn_out_hw=(64,64),
        frontcnn_pool='max',
        # CLSTM configuration
        clstm_hidden=32,
        clstm_kernel=3,
        # scalar LSTM configuration
        slstm_hidden=16,
        slstm_num_layers=1,
        slstm_dropout=0.0,
        # classifier head configuration
        classhead_hidden=64,
        classhead_dropout=0.0,
        classhead_activation='relu',
        # attention parameters
        attention_reduction=16,
        attention_kernel=7,
        # pooling before classification head mlp
        pool_type='avg'
    ):
        super(LakeDrainageClassifier, self).__init__()

        # store configuration
        self.use_imgseq = use_imgseq
        self.use_areaseq = use_areaseq
        self.use_cloudyseq = use_cloudyseq
        self.learn_area_weights = learn_area_weights
        self.learn_cloudy_weights = learn_cloudy_weights
        self.seq_len = seq_len
        self.use_nir = use_nir
        self.use_swir16 = use_swir16
        self.use_swir22 = use_swir22
        self.attention_type = attention_type.lower()
        self.pool_type = pool_type

        # Calculate number of imagery channels (RGB + optional spectral bands)
        # Mask is handled separately in the forward pass
        self.n_imagery_channels = 3  # RGB base
        if use_nir:
            self.n_imagery_channels += 1
        if use_swir16:
            self.n_imagery_channels += 1
        if use_swir22:
            self.n_imagery_channels += 1

        # validate attention type
        valid_attention = ['none', 'spatial', 'full', 'arch']
        if self.attention_type not in valid_attention:
            raise ValueError(f"Invalid attention_type '{attention_type}'. Must be one of {valid_attention}.")

        # == IMAGE SEQUENCE PROCESSING == #
        if use_imgseq:
            # (1) FrontCNN for imagery (RGB + optional spectral bands)
            self.frontcnn_rgb = FrontCNN(
                in_channels=self.n_imagery_channels,
                base_channels=frontcnn_base_channels,
                num_layers=frontcnn_num_layers,
                out_hw=frontcnn_out_hw,
                pool=frontcnn_pool,
            )
            frontcnn_out_channels = self.frontcnn_rgb.output_channels

            # (2) APPLY ATTENTION
            if self.attention_type == 'spatial':
                # spatial attention only
                self.attention = SpatialCBAM(
                    in_channels=frontcnn_out_channels,
                    kernel_size=attention_kernel,
                )
            elif self.attention_type == 'full':
                # full CBAM attention
                self.attention = FullCBAM(
                    in_channels=frontcnn_out_channels,
                    reduction_ratio=attention_reduction,
                    kernel_size=attention_kernel,
                )
            elif self.attention_type == 'arch':
                # architectural attention: separate pathway for lake mask
                self.frontcnn_mask = FrontCNN(
                    in_channels=1,
                    base_channels=frontcnn_base_channels,
                    num_layers=frontcnn_num_layers,
                    out_hw=frontcnn_out_hw,
                    pool=frontcnn_pool,
                )
            else: # no attention
                self.attention = nn.Identity()

            # (3) CLSTM
            self.clstm = CLSTM(
                input_channels=frontcnn_out_channels,
                hidden_channels=clstm_hidden,
                kernel_size=clstm_kernel,
                return_sequence=True,
            )

            # (4) GLOBAL POOLING
            self.global_pool = GlobalPooling(pool_type=pool_type)

        # == SCALAR TIME SEQUENCE PROCESSING == #
        # separate LSTM for each scalar sequence
        if use_areaseq:
            # Learnable per-timestep weights for area_seq
            if learn_area_weights:
                # Initialize to zeros so sigmoid gives 0.5 (neutral weighting)
                self.area_logits = nn.Parameter(torch.zeros(seq_len))

            self.area_lstm = ScalarLSTM(
                hidden_dim=slstm_hidden,
                num_layers=slstm_num_layers,
                dropout=slstm_dropout,
            )

        if use_cloudyseq:
            # Learnable per-timestep weights for cloudy_seq
            if learn_cloudy_weights:
                # Initialize to zeros so sigmoid gives 0.5 (neutral weighting)
                self.cloudy_logits = nn.Parameter(torch.zeros(seq_len))

            self.cloudy_lstm = ScalarLSTM(
                hidden_dim=slstm_hidden,
                num_layers=slstm_num_layers,
                dropout=slstm_dropout,
            )

        # == CLASSIFICATION HEAD == #
        # calculate input dimension for classificaiton head
        classifier_input_dim = 0
        if use_imgseq:
            # from CLSTM after global pooling
            if pool_type == 'both':
                classifier_input_dim += 2 * clstm_hidden
            else:
                classifier_input_dim += clstm_hidden
        if use_areaseq:
            classifier_input_dim += slstm_hidden
        if use_cloudyseq:
            classifier_input_dim += slstm_hidden
        
        if classifier_input_dim == 0:
            raise ValueError("At least one of use_imgseq, use_areaseq, or use_cloudyseq must be True.")

        # cloudyseq cannot be used alone; it requires imgseq or areaseq
        if use_cloudyseq and not (use_imgseq or use_areaseq):
            raise ValueError("use_cloudyseq cannot be enabled alone; it requires use_imgseq or use_areaseq to also be True.")

        self.classifier = ClassHeadMLP(
            input_dim=classifier_input_dim,
            hidden_dims=classhead_hidden,
            num_classes=num_classes,
            dropout=classhead_dropout,
            activation=classhead_activation,
        )

    def forward(self, x, area_seq, cloudy_seq):
        """
        Forward pass through the model.

        Args:
            x: [B, T, 4, H, W] tensor (RGB+mask) of image sequences
            area_seq: [B, T, 1] tensor of water area values
            cloudy_seq: [B, T, 1] tensor of cloud coverage values

        Returns:
            [B, num_classes] tensor of class logits
        """
        features = []

        # == IMAGE SEQUENCE PROCESSING == #
        if self.use_imgseq:
            # split the imagery channels and mask
            # Imagery is all channels except the last one (mask)
            imagery = x[:, :, :self.n_imagery_channels, :, :]  # [B, T, n_imagery_channels, H, W]
            mask = x[:, :, self.n_imagery_channels:, :, :]     # [B, T, 1, H, W]

            # process imagery through FrontCNN
            rgb_features = self.frontcnn_rgb(imagery)   # [B, T, C, Hf, Wf]

            # apply attention
            if self.attention_type == 'arch':
                # architectural attention fusion between separate mask pathway and RGB pathway
                mask_features = self.frontcnn_mask(mask)
                img_features = rgb_features * mask_features
            else:
                # learned attention (CBAM) or no attention
                img_features = self.attention(rgb_features)
            
            # CLSTM processing
            lstm_out = self.clstm(img_features)   # [B, T, C_hidden, Hf, Wf]

            # aggregate by taking last timestep and global pooling
            last_hidden = lstm_out[:, -1, :, :, :]   # [B, C_hidden, Hf, Wf]
            img_features = self.global_pool(last_hidden.unsqueeze(1)).squeeze(1)  # [B, C_hidden] or [B, 2*C_hidden]

            features.append(img_features)

        # == SCALAR TIME SEQUENCE PROCESSING == #
        # process lake area sequence
        if self.use_areaseq:
            # Apply learned temporal weights if enabled
            if self.learn_area_weights:
                T = area_seq.shape[1]
                area_weights = torch.sigmoid(self.area_logits[:T])  # [T]
                area_seq = area_seq * area_weights.view(1, -1, 1)  # [B, T, 1]

            area_out = self.area_lstm(area_seq) # [B, T, slstm_hidden]
            area_features = area_out[:, -1, :]  # [B, slstm_hidden]
            features.append(area_features)

        # process cloudy sequence
        if self.use_cloudyseq:
            # Apply learned temporal weights if enabled
            if self.learn_cloudy_weights:
                T = cloudy_seq.shape[1]
                cloudy_weights = torch.sigmoid(self.cloudy_logits[:T])  # [T]
                cloudy_seq = cloudy_seq * cloudy_weights.view(1, -1, 1)  # [B, T, 1]

            cloudy_out = self.cloudy_lstm(cloudy_seq) # [B, T, slstm_hidden]
            cloudy_features = cloudy_out[:, -1, :]  # [B, slstm_hidden]
            features.append(cloudy_features)

        # == CLASSIFICATION == #
        # concatenate all features
        combined_features = torch.cat(features, dim=1) # [B, total_dim]

        # classify
        logits = self.classifier(combined_features)  # [B, num_classes]

        return logits

    def get_feature_dims(self):
        """
        Get the dimensions of features from each component.
        
        Useful for debugging and understanding model architecture.
        
        Returns:
            dict: Feature dimensions for each component
        """
        dims = {}
        
        if self.use_imgseq:
            if self.pool_type == 'both':
                dims['img_features'] = self.clstm.hidden_channels * 2
            else:
                dims['img_features'] = self.clstm.hidden_channels
        
        if self.use_areaseq:
            dims['area_features'] = self.area_lstm.lstm.hidden_size
        
        if self.use_cloudyseq:
            dims['cloudy_features'] = self.cloudy_lstm.lstm.hidden_size
        
        dims['total'] = sum(dims.values())
        
        return dims
    
    def __repr__(self):
        """Custom string representation showing model configuration."""
        feature_dims = self.get_feature_dims()

        # Build spectral bands string
        spectral_bands = ['RGB']
        if self.use_nir:
            spectral_bands.append('NIR')
        if self.use_swir16:
            spectral_bands.append('SWIR16')
        if self.use_swir22:
            spectral_bands.append('SWIR22')
        spectral_str = '+'.join(spectral_bands)

        # Build area config string
        area_str = f"areaseq={self.use_areaseq}"
        if self.use_areaseq and self.learn_area_weights:
            area_str += " (learned weights)"

        # Build cloudy config string
        cloudy_str = f"cloudyseq={self.use_cloudyseq}"
        if self.use_cloudyseq and self.learn_cloudy_weights:
            cloudy_str += " (learned weights)"

        # Add seq_len if any learned weights are enabled
        seq_len_str = ""
        if (self.use_areaseq and self.learn_area_weights) or (self.use_cloudyseq and self.learn_cloudy_weights):
            seq_len_str = f", seq_len={self.seq_len}"

        config_str = (
            f"LakeDrainageClassifier(\n"
            f"  Features: "
            f"imgseq={self.use_imgseq}, "
            f"{area_str}, "
            f"{cloudy_str}{seq_len_str}\n"
            f"  Spectral bands: {spectral_str} ({self.n_imagery_channels} channels + mask)\n"
            f"  Attention: {self.attention_type}\n"
            f"  Feature dims: {feature_dims}\n"
            f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )
        return config_str
        
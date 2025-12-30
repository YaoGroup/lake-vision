"""
ClassifierLSTM module for Greenland supraglacial lake drainage classification.

Combines multiple components to classify supraglacial lake drainage events
from satellite imagery sequences and scalar time series data.
"""
import torch
import torch.nn as nn

from .blocks import FrontCNN, ScalarLSTM, ClassHeadMLP
from .clstm import CLSTM
from .attention import SpatialCBAM, FullCBAM

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
        x: [B, T, 4, H, W] tensor of image sequences
            B: batch size
            T: time steps
            4: input image channels (RGB + lake mask if attention_type='arch')
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
                attention_type='spatial',
            )
        >>> x = torch.randn(16, 153, 4, 512, 512)          # image sequences [B=16, T=153, C=4, H=512, W=512]
        >>> area_seq = torch.randn(16, 153, 1)              # water area sequences [B=16, T=153, 1]
        >>> logits = model(x, area_seq)                    # output logits [B=16, num_classes=4]
    """
    def __init__(
        self,
        # feature flags
        use_imgseq=True,
        use_areaseq=True,
        use_cloudyseq=False,
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
        self.attention_type = attention_type.lower()
        self.pool_type = pool_type

        # validate attention type
        valid_attention = ['none', 'spatial', 'full', 'arch']
        if self.attention_type not in valid_attention:
            raise ValueError(f"Invalid attention_type '{attention_type}'. Must be one of {valid_attention}.")
        
        # == IMAGE SEQUENCE PROCESSING == #
        if use_imgseq:
            # (1) FrontCNN for RGB
            self.frontcnn_rgb = FrontCNN(
                in_channels=3,
                base_channels=frontcnn_base_channels
                num_layers=frontcnn_num_layers,
                out_hw=frontcnn_out_hw,
                pool_type=frontcnn_pool,
            )
            frontcnn_out_channels = self.frontcnn_rgb.out_channels

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
                    pool_type=frontcnn_pool,
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
            self.area_lstm = ScalarLSTM(
                hidden_dim=slstm_hidden,
                num_layers=slstm_num_layers,
                dropout=slstm_dropout,
            )

        if use_cloudyseq:
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

        self.classifier = ClassHeadMLP(
            input_dim=classifier_input_dim,
            hidden_dim=classhead_hidden,
            output_dim=num_classes,
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
            # split the RGB and mask
            rgb = x[:, :, :3, :, :]         # [B, T, 3, H, W]
            mask = x[:, :, 3:, :, :]        # [B, T, 1, H, W]

            # process RGB through FrontCNN
            rgb_features = self.frontcnn_rgb(rgb)   # [B, T, C, Hf, Wf]

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
            area_out = self.area_lstm(area_seq) # [B, T, slstm_hidden]
            area_features = area_out[:, -1, :]  # [B, slstm_hidden]
            features.append(area_features)

        # process cloudy sequence
        if self.use_cloudyseq:
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
                dims['img_features'] = self.convlstm.hidden_channels * 2
            else:
                dims['img_features'] = self.convlstm.hidden_channels
        
        if self.use_areaseq:
            dims['area_features'] = self.area_lstm.lstm.hidden_size
        
        if self.use_cloudyseq:
            dims['cloudy_features'] = self.cloudy_lstm.lstm.hidden_size
        
        dims['total'] = sum(dims.values())
        
        return dims
    
    def __repr__(self):
        """Custom string representation showing model configuration."""
        feature_dims = self.get_feature_dims()
        
        config_str = (
            f"LakeDrainageClassifier(\n"
            f"  Features: "
            f"imgseq={self.use_imgseq}, "
            f"areaseq={self.use_areaseq}, "
            f"cloudyseq={self.use_cloudyseq}\n"
            f"  Attention: {self.attention_type}\n"
            f"  Feature dims: {feature_dims}\n"
            f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )
        return config_str
        
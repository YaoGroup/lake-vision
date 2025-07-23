

## CLASS FOR DATASET WITH IMAGE SEQUENCE AND LAKE AREA TIME CURVE
class DatasetImgseqAreacurve(Dataset):
    def __init__(self, fp_h5_images, ds_CW, seq_len=21, transform=None):
        self.fp_h5_images = fp_h5_images
        self.ds           = ds_CW
        self.seq_len      = seq_len
        self.transform    = transform

        self.area_seqs = ds_CW.S2_water.values # (n_lakes, n_time)
        self.idx_max = ds_CW.S2_water.argmax(dim="time").values # (n_lakes,)
        self.labels_Rines = ds_CW.label_Rines.values # (n_lakes,)
        self.labels_Dunmire = ds_CW.label_Dunmire.values # (n_lakes,)

    def __len__(self):
        return self.ds.sizes["ids"]

    def __getitem__(self, idx):
        # identify lake
        lakenum = self.ds.lakenum.values[idx]  # e.g. "2019cw_366"

        # load full image sequence
        with h5py.File(self.fp_h5_images, 'r') as f:
            img_seq = f[f'cw_2019_lakenum_{lakenum}'][()] # (T, C, H, W)

        # slice window around max‐area time
        center = int(self.idx_max[idx])
        half   = self.seq_len // 2
        start  = max(0, center - half)
        end    = min(img_seq.shape[0], center + half + 1)

        if end - start < self.seq_len:
            if start == 0:
                end = min(self.seq_len, img_seq.shape[0])
            else:
                start = max(0, img_seq.shape[0] - self.seq_len)

        img_seq = img_seq[start:end] # (seq_len, C, H, W)
        area    = self.area_seqs[idx, start:end] # (seq_len,)

        # labels
        label_Rines = int(self.labels_Rines[idx])
        label_Dunmire = int(self.labels_Dunmire[idx])

        # to torch tensors
        img_seq = torch.tensor(img_seq, dtype=torch.float32)
        area    = torch.tensor(area,   dtype=torch.float32).unsqueeze(-1)
        label_Rines   = torch.tensor(label_Rines,  dtype=torch.long)
        label_Dunmire   = torch.tensor(label_Dunmire,  dtype=torch.long)

        # optional transform
        if self.transform:
            img_seq = self.transform(img_seq)

        return img_seq, area, label_Rines, label_Dunmire, lakenum
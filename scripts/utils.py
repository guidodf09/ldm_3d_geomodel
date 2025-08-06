def build_hard_data_pickle(models, well_loc):
    models_T = models.transpose((0, 4, 3, 2, 1))
    
    for wn, (ix, iy) in well_loc.items():
        ix -= 1
        iy -= 1
        for iz in range(nz):
            data = models_T[:, iz, iy, ix, 0]
            if data.max() != data.min():
                print(f'Not all models have the same value at location {(ix, iy, iz)}', wn)

    well_hd_all = {}
    for wn, (ix, iy) in well_loc.items():
        ix -= 1
        iy -= 1
        well_hd_all[wn] = [(ix, iy, iz, models_T[0, iz, iy, ix, 0]) for iz in range(nz)]
        well_hd_all[wn] = np.array(well_hd_all[wn])
    
    return well_hd_all

def compute_hd_loss(y_pred, well_hd_all):
    ix = list(well_hd_all[:, 0].astype(int))
    iy = list(well_hd_all[:, 1].astype(int))
    iz = list(well_hd_all[:, 2].astype(int))
    
    v = torch.from_numpy(well_hd_all[:, -1]).float().to(device).repeat(y_pred.shape[0], 1)
    hd_loss = mae_loss(y_pred[:, 0, ix, iy, iz], v)
    
    return hd_loss

def model2tricat(model, thresh1, thresh2):
    
    model_copy = np.copy(model)
    model_copy[model_copy < thresh1] = 0.
    model_copy[(model_copy >= thresh1) & (model_copy <= thresh2)] = 0.5
    model_copy[model_copy > thresh2] = 1.
    
    return model_copy


# -*- coding: utf-8 -*-
# ---------------------

# this is a configuration file containing settings for
# (1) feature extractor (FX)
# (2) nowcasting model (NC)

# feature extractor settings
FX_LR = 0.0001  # learning rate used to trane the feature extractor
FX_N_WORKERS = 4  # worker(s) number of the dataloader
FX_BATCH_SIZE = 8  # batch size used to trane the feature extractor
FX_MAX_EPOCHS = 256  # maximum training duration (# epochs)
FX_PATIENCE = 16 # stop training if no improvement is seen for a ‘FX_PATIENCE’ number of epochs

# nowcasting classifier settings
NC_LR = 0.0001  # learning rate used to trane the nowcasting classifier
NC_N_WORKERS = 0  # worker(s) number of the dataloader
NC_BATCH_SIZE = 2  # batch size used to trane the nowcasting classifier
NC_MAX_EPOCHS = 256  # maximum training duration (# epochs)
NC_PATIENCE = 16 # stop training if no improvement is seen for a ‘NC_PATIENCE’ number of epochs
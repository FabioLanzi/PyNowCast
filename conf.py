# -*- coding: utf-8 -*-
# ---------------------

# this is a configuration file containing settings for
# (1) feature extractor (FX)
# (2) nowcasting model (NC)

# feature extractor settings
FX_LR = 0.0001  # learning rate used to trane the feature extractor
FX_N_WORKERS = 1  # worker(s) number of the dataloader
FX_BATCH_SIZE = 1  # batch size used to trane the feature extractor
FX_EPOCHS = 256  # TODO: delete this!

# nowcasting model settings
NC_LR = 0.0001  # learning rate used to trane the nowcasting model
NC_N_WORKERS = 1  # worker(s) number of the dataloader
NC_BATCH_SIZE = 1  # batch size used to trane the nowcasting model
NC_EPOCHS = 256  # TODO: delete this!

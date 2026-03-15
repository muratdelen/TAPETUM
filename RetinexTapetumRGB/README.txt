RetinexTapetumRGB standardized project files

This variant uses:
- classical Gaussian Retinex decomposition,
- RGB Tapetum attention,
- learnable RGB channel gates,
- sigmoid-bounded lambda,
- enhanced illumination smoothness regularization.

Main updates applied to this version:
- added detailed explanations to all files,
- standardized config/train/test structure,
- changed lambda from softplus to bounded sigmoid:
      lambda = lambda_max * sigmoid(lambda_param)
- added edge-aware smoothness loss on illumination_t,
- changed attention regularization to a more stable form,
- added early stopping,
- added gradient clipping,
- fixed history/checkpoint write order,
- added history.csv export for easier comparison with other models.

Expected structure in Colab:
/content/TAPETUM/RetinexTapetumRGB/
    config.py
    dataset.py
    losses.py
    model.py
    train.py
    test.py
    utils.py

Run:
%cd /content/TAPETUM/RetinexTapetumRGB
!python train.py
!python test.py

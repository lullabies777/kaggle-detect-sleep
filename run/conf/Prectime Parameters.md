# feature extractor
- [ ] left_hidden_channels: list[int]
- [ ] right_hidden_channels: list[int]
- [ ] left_fe_kernel_size: int
- [ ] right_fe_kernel_size: int
- [ ] fe1_layers: int
- [ ] fe2_layers: int

# encoder
- [ ] fe_fc_dimension: int # flaten 后接的dense层的output shape
- [ ] lstm_dimensions: list[int]
- [ ] num_layers: int
# decoder
- [ ] out_channels: int
- [ ] kernel_size: int
- [ ] mode: str

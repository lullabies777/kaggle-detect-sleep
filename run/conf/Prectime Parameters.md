# feature extractor
- [ ] input_channels: int  # 表示输入的每个step的特征个数
- [ ] left_hidden_channels: list[int]  # 表示左侧channel数的变化，list中也包含了第一步的channel
- [ ] right_hidden_channels: list[int]
- [ ] left_fe_kernel_size: int # 表示左侧kernel的变化
- [ ] right_fe_kernel_size: int
- [ ] left_fe_padding: int # 表示左侧padding的变化，不一定要输入，可自动调节
- [ ] right_fe_padding: int 
- [ ] left_fe_stride: int # 表示左侧stride的变化，基本锁死都为1
- [ ] right_fe_stride: int
- [ ] left_fe_dilation: int # 表示左侧dilation的变化
- [ ] right_fe_dilation: int
- [ ] chunks: int, # 表示总输入被分为几块
- [ ] num_left_fe_layers: int # 表示左侧conv1d的总层数，不一定要输入，可自动调节
- [ ] num_right_fe_layers: int

# encoder
- [ ] fe_fc_dimension: int   # flatten 后接的dense层的output shape

  ##### 如果使用LSTM

- [ ] lstm_dimensions: list[int] # 表示每一层lstm的hidden_size

- [ ] num_lstm_layers=2 # 表示总共lstm的层数，不一定要输入，可自动调节

  ##### 如果使用transformer

- [ ] n_head: int # 表示transformer中有几个注意力头

- [ ] num_encoder_layers: int

- [ ] dim_feedforward: int # 表示其中mlp层的hidden_size

- [ ] dropout: float

- [ ] activation: string # 表示全连接层后使用的激活函数

  

- [ ] encoder_type: string # 表示是用transformer还是lstm

# decoder
- [ ] out_channels: int # 这里定死为128
- [ ] kernel_size: int # 都使用同一个kernel_size
- [ ] padding: int
- [ ] stride: int # 默认为1
- [ ] dilation: int
- [ ] mode: str

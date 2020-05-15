import torch
from torch import nn

class Encoder(nn.Module):

    def __init__(self, device, trg_pad_idx, kernel_size = 3, num_layers = 9, hidden_dim= 512, input_dimension=128, embedding_dimension= 128, max_length = 100, dropout = 0.25):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dimension = input_dimension
        self.max_length = max_length
        self.trg_pad_idx = trg_pad_idx
        enc_layers = []

        for i in range(self.num_layers):
            enc_layers.append(nn.Conv1d(in_channels = self.hidden_dim,
                                              out_channels = 2 * self.hidden_dim,
                                              kernel_size = self.kernel_size,
                                              padding = (self.kernel_size - 1)// 2))

        self.encoder = nn.Sequential(*enc_layers)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.device = device
        self.embedding_dim = embedding_dimension
        self.word_embedding = nn.Embedding(self.input_dimension,self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_projection = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.convolution_projection = nn.Linear(self.hidden_dim, self.embedding_dim)

    def forward(self, src):
        #src = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        word_embedding = self.word_embedding(src)
        position_embedding = self.position_embedding(pos)
        embedding = word_embedding + position_embedding
        embedding = self.dropout(embedding)
        #project embedding to conv dimensions
        conv_embedding = self.embedding_projection(embedding)
        conv_embedding = conv_embedding.permute(0,2,1)

        for layer in self.encoder.children():
            x = self.dropout(conv_embedding)

            x = layer(x)

            # Gate
            x = nn.functional.glu(x, dim=1)

            #Residual block . Add input of each convolution.
            x = x + conv_embedding
            #Normalize
            x = x * self.scale

            conv_embedding = x

        #project to embedding dim
        conv_embedding = conv_embedding.permute(0,2,1)
        embed_conv = self.convolution_projection(conv_embedding)
        out = embed_conv + embedding
        out = out * self.scale

        return embed_conv , out

class Decoder(nn.Module):
    def __init__(self, device, trg_pad_idx,out_dim = 4, kernel_size = 5, num_layers = 2 ,hidden_dim= 512, input_dimension=128, embedding_dimension= 128):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dimension = input_dimension
        self.trg_pad_idx = trg_pad_idx
        dec_layers = []
        for i in range(self.num_layers):
            dec_layers.append(nn.Conv1d(in_channels = self.hidden_dim,
                                              out_channels = 2 * self.hidden_dim,
                                              kernel_size = self.kernel_size))

        self.encoder = nn.Sequential(*dec_layers)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.device = device
        self.embedding_dim = embedding_dimension
        self.out_dim = out_dim

        self.word_embedding = nn.Embedding(self.out_dim,self.embedding_dim)
        self.position_embedding = nn.Embedding(self.hidden_dim, self.embedding_dim)
        self.dropout = nn.Dropout(0.25)
        self.embedding_projection = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.convolution_projection = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.attentionLinearcomb = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.embedding_dim, self.out_dim)
        self.previousCast = nn.Linear(self.out_dim, self.embedding_dim)
    def forward(self, target, embed_conv, encoder_output, previous_embedded_target):
        # src = [batch size, src len]
        batch_size = target.shape[0]
        target_len = target.shape[1]
        # create position tensor
        pos = torch.arange(0, target_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        word_embedding = self.word_embedding(target)
        position_embedding = self.position_embedding(pos)
        embedding = word_embedding + position_embedding
        embedding = self.dropout(embedding)
        # project embedding to conv dimensions
        conv_embedding = self.embedding_projection(embedding)

        conv_embedding = conv_embedding.permute(0,2,1)

        for layer in self.encoder.children():
            padding = torch.zeros(batch_size, self.hidden_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, self.dropout(conv_embedding)), dim=2)
            x = layer(padded_conv_input)
            #Gate
            x = nn.functional.glu(x, dim=1)
            #Residual block . Add input of each convolution.
            x = x + conv_embedding

            #Multi-step Attention

            #d_l^i = W_d^l h_l^i + b_l^d + g_i
            #Score : d_l^i dot z_j^u where z_j^u the output of the last encoderblock u:

            #Dot product with every state of previous encoder block

            projected_encoder_conv = self.embedding_projection(embed_conv)


            attention_weight = projected_encoder_conv.matmul(x)
            # attention_weight shape : [batch_size, src_len, target_len]
            attention_weight = nn.functional.softmax(attention_weight, dim=2)

            attention_weight = attention_weight.permute(0,2,1)
            attention = attention_weight.matmul(encoder_output)
            # attention shape : [batch_size, target_len, embed_dim]


            #project to hidden_dim dimension
            attention = self.embedding_projection(attention)
            attention = attention.permute(0,2,1)

            x = x + attention

            # Residual block . Add input of each convolution.
            x = x + conv_embedding

            # Normalize
            x = x * self.scale

            conv_embedding = x


        # project to hidden dim

        conv_embedding = conv_embedding.permute(0, 2, 1)
        conv_embedding = self.convolution_projection(conv_embedding)

        conv_embedding = self.dropout(conv_embedding)


        out = self.fc_out(conv_embedding)

        return out, attention


class Conv2Seq(nn.Module):
    def __init__(self, device, target_pad_index, enc_kernel_size = 3, dec_kernel_size = 5, enc_num_layers= 9,dec_num_layers= 2, hidden_dim= 512, input_dimension =128, out_dim=2):
        super().__init__()
        self.input_dim = input_dimension
        self.enc_kernel = enc_kernel_size
        self.dec_kernel = dec_kernel_size
        self.hidden_dim = hidden_dim
        self.enc_layers = enc_num_layers
        self.dec_layers= dec_num_layers
        self.encoder = Encoder(device = device, trg_pad_idx= target_pad_index, input_dimension = self.input_dim)
        self.decoder = Decoder(device = device, trg_pad_idx= target_pad_index, input_dimension= self.input_dim, out_dim= out_dim)
        self.previous_token = None

    def forward(self, src , trg):
        enc_conv_out, out = self.encoder(src )

        output, attention = self.decoder(trg, enc_conv_out, out, self.previous_token)

        return output, attention
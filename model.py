import torch.nn as nn
import math 
import numpy as np
import torch



class InputEmbedding(nn.Module) :
    def __init__(self, d_model: int, vocab_size: int)-> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)  # remove this for decoder side  


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int , seq_len: int, dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) 

        # create matrix of matrix (seq_len, d_model)

        pe = torch.zeros(seq_len, d_model)

        #create a vector of shape seq_len, 1
        position = torch.arange(0, seq_len, dtype= torch.float).unsqueeze(1)

        # divide the vector by 10000
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        # apply sign to even positions
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x ):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6 ) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added 


    def forward(self, x):
        mean = x.mean(dim = -1, keepdim  = True)
        std = x.std(dim = -1, keepdim  = True)
        return self.alpha * (x-mean)/ (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B!
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W@ amd B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (batch, seq_len, d_ff ) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int,dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, 'd_model is not divisible by h'
        self.d_k = d_model // h

        self.w_q  = nn.Linear(d_model, d_model)
        self.w_k  = nn.Linear(d_model, d_model) 
        self.w_v  = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k ) --> (Batch, h, seq_len, seq_len )    
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h , seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return(attention_scores @ value), attention_scores    



    def forward(self, q, k, v, mask ):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)


        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)


        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h , d_k) --> (Batch, seq_len, d_model)
        x=  x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm  = LayerNormalization()

    def forward(self, x, sublayer):
        return x +  self.dropout(sublayer(self.norm(x)))   


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()     
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[0](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()

    def forward(self, x, mask ):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  

class  DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,dropout: float ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.self_feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, tgt_mask):    
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x,self.self_feed_forward_block )
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        # (Batch, seq_len, d_model) --> (batch, seq_len, vocab_size)

        return torch.log_softmax(self.proj(x), dim=-1)



class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed  = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos  = tgt_pos
        self.projection_layer = projection_layer
    def encode( self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    


def build_transformer(src_vocab_size: int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model: int = 512, N:int = 6, h: int = 8, dropout: float = 0.1, d_ff:int = 2048 ) -> Transformer:
    # create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    #create postional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #create encdoer block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    # create encdoer and decoder
    encoder  =  Encoder(nn.ModuleList(encoder_blocks))    
    decoder  =  Decoder(nn.ModuleList(decoder_blocks)) 

    #create the projection_layer
    projecton_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projecton_layer)

    #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer        





















        
         








       




# class multiHeadAttention (nn.module):
#      def __init__(self, d_model, num_heads, key, query, value, seq_len):
#          super(multiHeadAttention, self).__init__()
#          self.d_model = d_model
#          self.num_h`eads = num_heads
#          self.depth = d_model // num_heads
#          self.seq_len = seq_len
#          self.key   = key
#          self.query = query
#          self.value = value


#     #divide each key, query, value matrice across each head 
     
          
#      def split_heads(self, x):
#             batch_size = x.shape[0]

#             split_inputs = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
#             return tf.transpose(split_inputs, perm=[0, 2, 1, 3])

#      def merge_heads(self, x):
#             batch_size = x.shape[0]

#             merged_inputs = tf.transpose(x, perm=[0, 2, 1, 3])
#             return tf.reshape(merged_inputs, (batch_size, -1, self.d_model))
     
#      def dot_product(self, q_s, k_s, v_s):
#             qk = tf.matmul(q_s, k_s, transpose_b=True)
#             dk = tf.cast(tf.shape(k_s)[-1], tf.float32)
#             scaled_attention_logits = qk / tf.math.sqrt(dk)
#             attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
#             output = tf.matmul(attention_weights, v_s)
#             return output, attention_weights


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super(PositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.max_seq_len = max_seq_len
        

#     def forward(self, x):
#         pos_encoding = np.array([
#             [pos / np.power(10000, 2.0 * (j // 2) / self.d_model) for j in range(self.d_model)]
           
             
         
         




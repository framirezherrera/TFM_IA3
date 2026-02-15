# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: TFM_Classic.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Classic Transformer
# ----------------------------------------------------------------------------------------------------------------------------------
# Implementado por: Felipe Ramírez Herrera
# Trabajo Final de Máster (TFM)
# Master de Inteligencia Artificial Avanzada y Aplicada (IA3)
# Universidad de Valencia / ADEIT
# Última revisión: 30/07/2024 
# ----------------------------------------------------------------------------------------------------------------------------------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
import copy

# Generación de máscaras para modelos basados en transformers.

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_padding_mask(src, padding_index):
    return (src == padding_index).transpose(0, 1)

def create_mask(src, tgt, src_padding_index, tgt_padding_index, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = get_padding_mask(src, src_padding_index)
    tgt_padding_mask = get_padding_mask(tgt, tgt_padding_index)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Este código implementa un módulo de codificación posicional para su uso en modelos de aprendizaje 
# profundo, como redes neuronales, particularmente en aplicaciones de procesamiento de lenguaje natural (NLP). 
# La codificación posicional es crucial en estos modelos para proporcionar información sobre la posición de 
# las palabras en una secuencia.

# Basado en https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py

# Se definen tres parámetros:

#   emb_size: El tamaño de las incrustaciones de tokens (embedding) y también el tamaño de las dimensiones de la codificación posicional.
#   dropout: La tasa de abandono (dropout) que se aplicará a la salida.
#   maxlen: La longitud máxima de la secuencia para la cual se calculará la codificación posicional. Por defecto, se establece en 5000.

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # Se crea un tensor den que contiene los valores de exponenciación negativa necesarios para calcular los términos seno y coseno.
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        # Se crea un tensor pos que contiene los índices de posición de la secuencia de longitud maxlen.
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # Se crea un tensor pos_embedding inicializado con ceros, de tamaño (maxlen, emb_size), para almacenar la codificación posicional.
        pos_embedding = torch.zeros((maxlen, emb_size))
        # Se calculan y asignan los valores de los términos seno y coseno a pos_embedding.
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        # Finalmente, pos_embedding se expande en una dimensión adicional para permitir la transmisión con otros tensores de entrada.
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # Se aplica dropout a token_embedding.
        # Se suma token_embedding con la codificación posicional correspondiente. Solo se toman las primeras token_embedding.size(0) filas de pos_embedding, que coinciden con la longitud de la secuencia de entrada.
        # La salida es el tensor resultante después de aplicar el dropout y sumar la codificación posicional.
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


    

# Un módulo para convertir un tensor de índices de entrada en un tensor correspondiente de embedings de tokens
# Este código define una clase llamada TokenEmbedding que se utiliza para generar representaciones vectoriales 
# (también conocidas como embeddings) para tokens.      
# Basado en https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py

class TokenEmbedding(nn.Module):

    # Este método inicializa la instancia de la clase. Toma dos parámetros: vocab_size, que es el tamaño del 
    # vocabulario (es decir, la cantidad de tokens distintos en el corpus), y emb_size, que es el tamaño de 
    # los vectores de embedding que se generarán para cada token.

    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        #  Crea una capa de embedding utilizando nn.Embedding, que asigna un vector de embedding único a cada 
        # token en el rango [0, vocab_size).
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    # Define el método forward, que especifica cómo se calculará la salida de la red cuando se pasa una entrada 
    # a través del modelo.
    def forward(self, tokens):
        # Se multiplica cada embedding resultante por la raíz cuadrada del tamaño del embedding (emb_size). Esto se
        # hace para escalar los embeddings y prevenir que los valores sean demasiado pequeños o grandes, lo que puede 
        # afectar el proceso de entrenamiento.
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
    def shared_weights_with(self, weight):
        # Define un método llamado shared_weights_with, que permite compartir los pesos de la capa de embedding con 
        # otra capa de embedding.
        self.embedding.weight = weight



#    Implementa una capa personalizada del encoder para un Transformer.
#
#    Parámetros:
#        d_model (int): Tamaño del espacio de representación.
#        head_num (int): Número de cabezas en la atención múltiple.
#        ff_dim (int, opcional): Dimensión de la capa oculta en la red de alimentación hacia adelante.
#        dropout (float, opcional): Probabilidad de dropout.
#        layer_norm_eps (float, opcional): Epsilon para la normalización de capa.
#        batch_first (bool, opcional): Si la entrada y salida están en formato de lote primero.
#        norm_first (bool, opcional): Si se aplica la normalización de capa antes de las operaciones de atención y alimentación hacia adelante.
#        bias (bool, opcional): Si se incluyen sesgos en las capas lineales.
#        device (str o dispositivo, opcional): Dispositivo de cómputo para los parámetros de la capa.
#        dtype (torch.dtype, opcional): Tipo de datos para los parámetros de la capa.

#    Métodos:
#        forward(src, src_mask, src_key_padding_mask):
#            Realiza un paso hacia adelante de la capa del encoder personalizada.

#        _sa_block(x, attn_mask, key_padding_mask):
#            Realiza la parte de la atención propia de la capa.

#        _ff_block(x):
#            Realiza la parte de la red de alimentación hacia adelante de la capa.

# Basado en https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py 

class CustomTransformerEncoderLayer(nn.Module):
    
    # Inicializa una capa personalizada del encoder para un Transformer.

    def __init__(self, d_model: int, head_num: int, ff_dim: int = 2048, dropout: float = 0.1, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, head_num, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, ff_dim, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    # Realiza un paso hacia adelante de la capa del encoder personalizada.
    #
    #    Parámetros:
    #        src (torch.Tensor): Tensor de entrada.
    #        src_mask (torch.BoolTensor): Máscara opcional para evitar la atención sobre ciertas posiciones.
    #        src_key_padding_mask (torch.BoolTensor): Máscara opcional para evitar la atención sobre ciertas posiciones basadas en la entrada.

    #    Retorna:
    #        torch.Tensor: Tensor de salida.

    def forward(self,  src,  src_mask, src_key_padding_mask):
        src_key_padding_mask = F._canonical_mask(mask=src_key_padding_mask, mask_name="src_key_padding_mask", other_type=F._none_or_dtype(src_mask), other_name="src_mask", target_type=src.dtype)
        src_mask = F._canonical_mask(mask=src_mask, mask_name="src_mask", other_type=None, other_name="", target_type=src.dtype, check_other=False)
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


    # Realiza la parte de la self-attention de la capa.

    #    Parámetros:
    #        x (torch.Tensor): Tensor de entrada.
    #        attn_mask (torch.BoolTensor): Máscara opcional para evitar la atención sobre ciertas posiciones.
    #        key_padding_mask (torch.BoolTensor): Máscara opcional para evitar la atención sobre ciertas posiciones basadas en la entrada.

    #    Retorna:
    #        torch.Tensor: Tensor de salida después de la atención propia.

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, is_causal=False)[0]
        return self.dropout1(x)

    # Realiza la parte de la red de alimentación hacia adelante de la capa.

    #    Parámetros:
    #        x (torch.Tensor): Tensor de entrada.

    #    Retorna:
    #        torch.Tensor: Tensor de salida después de la red de alimentación hacia adelante.

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# Implementa un encoder personalizado para un Transformer.
# Basado en https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py

#    Parámetros:
#        encoder_layer (nn.Module): Capa de encoder a utilizar en cada paso.
#        num_layers (int): Número de capas de encoder.
#        norm (nn.Module, opcional): Capa de normalización a aplicar después de todas las capas de encoder.

#    Métodos:
#        forward(src, mask=None, src_key_padding_mask=None):
#            Realiza un paso hacia adelante a través del encoder personalizado.

#    Atributos:
#        layers (list): Lista de capas de encoder clonadas.
#        num_layers (int): Número de capas de encoder.
#        norm (nn.Module): Capa de normalización.
    

class CustomTransformerEncoder(nn.Module):

    # Inicializa un encoder personalizado para un Transformer.
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    # Realiza un paso hacia adelante a través del encoder personalizado.

    #   Parámetros:
    #        src (torch.Tensor): Tensor de entrada.
    #        mask (torch.BoolTensor, opcional): Máscara opcional para evitar la atención sobre ciertas posiciones.
    #        src_key_padding_mask (torch.BoolTensor, opcional): Máscara opcional para evitar la atención sobre ciertas posiciones basadas en la entrada.

    #    Retorna:
    #        torch.Tensor: Tensor de salida.

    def forward(self, src, mask = None, src_key_padding_mask = None):   
        src_key_padding_mask = F._canonical_mask(mask=src_key_padding_mask, mask_name="src_key_padding_mask", other_type=F._none_or_dtype(mask), other_name="mask", target_type=src.dtype)
        mask = F._canonical_mask(mask=mask, mask_name="mask", other_type=None, other_name="", target_type=src.dtype, check_other=False)
        x = src
        for layer in self.layers:
            x = layer(x, mask, src_key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


    
# Esta clase CustomTransformerDecoderLayer es una capa personalizada para el decodificador de un modelo de Transformer. 
# Esta clase está diseñada para ser parte de un decodificador en un modelo Transformer, como los utilizados en tareas de
# procesamiento de lenguaje natural, como la traducción automática.
# Basado en https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py

class CustomTransformerDecoderLayer(nn.Module):

    # Esta función se llama cuando se instancia un objeto de la clase CustomTransformerDecoderLayer.

    # Los parámetros de entrada son:
    #   d_model: Dimensión del modelo. Es el tamaño de representación en el espacio de embedding.
    #   head_num: Número de cabezas de atención en la capa de atención múltiple.
    #   ff_dim: Dimensión de la capa de feedforward.
    #   dropout: Tasa de dropout para las capas de dropout.
    #   layer_norm_eps: Valor epsilon para la normalización por capa.
    #   batch_first: Booleano que indica si la dimensión del lote es la primera dimensión de los datos de entrada.
    #   norm_first: Booleano que indica si la normalización se aplica antes o después de las operaciones en la capa.
    #   bias: Booleano que indica si se incluye sesgo en las capas lineales.
    #   device: Dispositivo en el que se realizarán los cálculos.
    #   dtype: Tipo de datos de los tensores.

    def __init__(self, d_model: int, head_num: int, ff_dim: int = 2048, dropout: float = 0.1, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,  bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, head_num, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, head_num, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs)
        self.fc1 = nn.Linear(d_model, ff_dim, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    # Este método define cómo se calcula la salida de la capa.
    #   Los parámetros de entrada son:
    #   tgt: Datos de entrada para la capa.
    #   memory: Datos de memoria (salida del codificador) para la atención.
    #   tgt_mask: Máscara para ocultar tokens futuros en tgt.
    #   memory_mask: Máscara para ocultar tokens en memory.
    #   tgt_key_padding_mask: Máscara para ocultar elementos de relleno en tgt.
    #   memory_key_padding_mask: Máscara para ocultar elementos de relleno en memory.

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        # Primero, se copia el tensor de entrada tgt a la variable x.
        x = tgt
        # Luego, dependiendo del valor de norm_first, se realiza un paso diferente:
        # Si norm_first es True, la normalización se aplica antes de cada paso de la capa.
        # Si norm_first es False, la normalización se aplica después de cada paso de la capa.
        # En cada paso, se realiza la siguiente operación:
        # - Se aplica atención propia (self-attention) o atención múltiple con respecto a la memory.
        # - Se añade una capa de feedforward seguida de una activación ReLU.
        # - Se aplica dropout.
        # - Se añade la salida a x y se normaliza si es necesario.
        # - Finalmente, se devuelve x, que es la salida de la capa después de pasar por todos los pasos.  

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x

    # Estos métodos _sa_block, _mha_block y _ff_block son bloques de operaciones que forman parte de la clase CustomTransformerDecoderLayer. 
    # Cada uno de estos métodos representa una etapa específica en el procesamiento de datos dentro de una capa del decodificador Transformer.

    # self-attention block
    # Este método realiza la operación de self-attention en los datos de entrada x.
    # Utiliza el módulo de self_attn, que es una capa de atención múltiple definida en el constructor.
    # Los parámetros de entrada son x, attn_mask y key_padding_mask.
    # Se aplica autoatención utilizando x como consulta, clave y valor.
    # attn_mask se utiliza para enmascarar posiciones futuras en la secuencia de entrada.
    # key_padding_mask se utiliza para enmascarar posiciones de relleno en la secuencia de entrada.
    # Después de aplicar la atención, se aplica una capa de dropout (self.dropout1) a la salida.
    # La salida se devuelve como resultado del bloque de autoatención.

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,is_causal=False, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    # Este método realiza la operación de atención múltiple entre los datos de entrada x y los datos de memoria mem.
    # Utiliza el módulo multihead_attn, que también es una capa de atención múltiple definida en el constructor.
    # Los parámetros de entrada son x, mem, attn_mask y key_padding_mask.
    # Se aplica atención múltiple utilizando x como consulta y mem como clave y valor.
    # attn_mask y key_padding_mask se utilizan de manera similar a _sa_block.
    # Después de aplicar la atención múltiple, se aplica una capa de dropout (self.dropout2) a la salida.
    # La salida se devuelve como resultado del bloque de atención múltiple.

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=False, need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    # Este método realiza la operación de feedforward en los datos de entrada x.
    # Utiliza dos capas lineales (self.fc1 y self.fc2) seguidas de una función de activación ReLU (self.activation).
    # Los parámetros de entrada son x.
    # Se aplica una capa de dropout (self.dropout) después de la primera capa lineal.
    # La salida de la segunda capa lineal se devuelve como resultado del bloque de feedforward después de aplicar una última capa de dropout (self.dropout3).

    def _ff_block(self, x):
        x = self.fc2(self.dropout(self.activation(self.fc1(x))))
        return self.dropout3(x)

# La clase CustomTransformerDecoder representa el decodificador completo de un modelo Transformer personalizado. 
# Su objetivo es aplicar múltiples capas de decodificador (representadas por decoder_layer) de manera secuencial.
# Basado en https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
    
class CustomTransformerDecoder(nn.Module):
    # Esta función se llama cuando se instancia un objeto de la clase CustomTransformerDecoder.
    
    # Los parámetros de entrada son:
    #   - decoder_layer: La capa del decodificador que se utilizará en cada paso.
    #   - num_layers: Número de capas de decodificador que se apilarán.
    #   - norm: Capa de normalización opcional que se aplicará al resultado final.
    
    # Se inicializan los siguientes atributos:
    #   self.layers: Una lista que contiene las capas de decodificador clonadas (decoder_layer) según el número de capas especificado.
    #   self.num_layers: El número total de capas de decodificador.
    #   self.norm: La capa de normalización opcional que se aplicará al resultado final. Si no se proporciona, se espera que sea None.

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    # Este método define cómo se calcula la salida del decodificador.
    # Los parámetros de entrada son:
    #   tgt: Datos de entrada para el decodificador.
    #   memory: Datos de memoria (salida del codificador) para la atención.
    #   tgt_mask: Máscara para ocultar tokens futuros en tgt.
    #   memory_mask: Máscara para ocultar tokens en memory.
    #   tgt_key_padding_mask: Máscara para ocultar elementos de relleno en tgt.
    #   memory_key_padding_mask: Máscara para ocultar elementos de relleno en memory.
    # El método itera sobre todas las capas de decodificador almacenadas en self.layers.
    # En cada iteración, aplica una capa de decodificador a los datos de entrada junto con las máscaras necesarias.
    # Después de pasar por todas las capas de decodificador, si se proporciona una capa de normalización (self.norm), se aplica a la salida final.
    # La salida final del decodificador se devuelve como resultado.

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

# La clase CustomSingleDecoderTransformer es un modelo de Transformer personalizado que consta de un codificador y un decodificador. 
# Basado en https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
    
class CustomSingleDecoderTransformer(nn.Module):

    # Esta función se llama cuando se instancia un objeto de la clase CustomSingleDecoderTransformer.
    # Los parámetros de entrada son:
    #   d_model: Dimensión del modelo. Es el tamaño de representación en el espacio de embedding.
    #   head_num: Número de cabezas de atención en las capas de atención.
    #   num_encoder_layers: Número de capas en el codificador.
    #   num_decoder_layers: Número de capas en el decodificador.
    #   ff_dim: Dimensión de la capa de feedforward.
    #   dropout: Tasa de dropout para las capas de dropout.
    #   layer_norm_eps: Valor epsilon para la normalización por capa.
    #   batch_first: Booleano que indica si la dimensión del lote es la primera dimensión de los datos de entrada.
    #   norm_first: Booleano que indica si la normalización se aplica antes o después de las operaciones en las capas.
    #   bias: Booleano que indica si se incluye sesgo en las capas lineales.
    #   device: Dispositivo en el que se realizarán los cálculos.
    #   dtype: Tipo de datos de los tensores.
    #   Se inicializan el codificador y el decodificador, cada uno con sus propias capas de normalización y capas de Transformer específicas.
    #   Se almacenan las dimensiones importantes d_model, head_num y batch_first.

    def __init__(self, d_model = 512, head_num = 8, num_encoder_layers = 6, num_decoder_layers = 6, ff_dim = 2048, dropout = 0.1,
                 layer_norm_eps = 1e-5, batch_first = False, norm_first = False,
                 bias = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        encoder_layer = CustomTransformerEncoderLayer(d_model, head_num, ff_dim, dropout,layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = CustomTransformerDecoderLayer(d_model, head_num, ff_dim, dropout, layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder = CustomTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.d_model = d_model
        self.head_num = head_num
        self.batch_first = batch_first

    # Este método define cómo se calcula la salida del modelo.
    
    # Los parámetros de entrada son:
    #   src: Datos de entrada para el codificador.
    #   tgt: Datos de entrada para el decodificador.
    #   src_mask: Máscara para ocultar tokens en src.
    #   tgt_mask: Máscara para ocultar tokens futuros en tgt.
    #   memory_mask: Máscara para ocultar tokens en la memoria (salida del codificador).
    #   src_key_padding_mask: Máscara para ocultar elementos de relleno en src.
    #   tgt_key_padding_mask: Máscara para ocultar elementos de relleno en tgt.
    #   memory_key_padding_mask: Máscara para ocultar elementos de relleno en la memoria.

    # Verifica si los datos de entrada tienen la forma correcta y el tamaño adecuado.
    # Pasa los datos de entrada a través del codificador y almacena la salida en memory.
    # Luego, pasa los datos de entrada del decodificador junto con memory a través del decodificador.
    # La salida final del decodificador se devuelve como resultado.

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):     
        is_batched = src.dim() == 3
        if src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output


# La clase CustomDoubleDecoderTransformer es un modelo de Transformer personalizado que consta de un codificador y dos decodificadores. 
# Cada decodificador se utiliza para generar salidas independientes y puede ser útil en tareas donde se requieren múltiples secuencias de salida. 

class CustomDoubleDecoderTransformer(nn.Module):

    # Esta función se llama cuando se instancia un objeto de la clase CustomDoubleDecoderTransformer.
    
    # Los parámetros de entrada son similares a los del modelo de un solo decodificador, con algunas diferencias notables:
    #   d_model: Dimensión del modelo. Es el tamaño de representación en el espacio de embedding.
    #   head_num: Número de cabezas de atención en las capas de atención.
    #   num_encoder_layers: Número de capas en el codificador.
    #   num_decoder_layers: Número de capas en cada decodificador.
    #   ff_dim: Dimensión de la capa de feedforward.
    #   dropout: Tasa de dropout para las capas de dropout.
    #   layer_norm_eps: Valor epsilon para la normalización por capa.
    #   batch_first: Booleano que indica si la dimensión del lote es la primera dimensión de los datos de entrada.
    #   norm_first: Booleano que indica si la normalización se aplica antes o después de las operaciones en las capas.
    #   bias: Booleano que indica si se incluye sesgo en las capas lineales.
    #   device: Dispositivo en el que se realizarán los cálculos.
    #   dtype: Tipo de datos de los tensores.

    # Se inicializan el codificador y dos decodificadores, cada uno con sus propias capas de normalización y capas de Transformer específicas.
    # Se almacenan las dimensiones importantes d_model, head_num y batch_first.

    def __init__(self, d_model = 256, head_num = 8, num_encoder_layers = 6, num_decoder_layers = 6, ff_dim = 512, dropout = 0.1, layer_norm_eps = 1e-5, batch_first = False, norm_first = False,
                 bias = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        encoder_layer = CustomTransformerEncoderLayer(d_model, head_num, ff_dim, dropout, layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = CustomTransformerDecoderLayer(d_model, head_num, ff_dim, dropout, layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder_a = CustomTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        decoder_layer = CustomTransformerDecoderLayer(d_model, head_num, ff_dim, dropout, layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder_b = CustomTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
      
        self.d_model = d_model
        self.nhead = head_num
        self.batch_first = batch_first

    # Este método define cómo se calcula la salida del modelo.

    # Los parámetros de entrada son:
    #   src: Datos de entrada para el codificador.
    #   tgt_a: Datos de entrada para el primer decodificador.
    #   tgt_b: Datos de entrada para el segundo decodificador.
    #   src_mask: Máscara para ocultar tokens en src.
    #   tgt_a_mask: Máscara para ocultar tokens futuros en tgt_a.
    #   tgt_b_mask: Máscara para ocultar tokens futuros en tgt_b.
    #   memory_mask: Máscara para ocultar tokens en la memoria (salida del codificador).
    #   src_key_padding_mask: Máscara para ocultar elementos de relleno en src.
    #   tgt_a_key_padding_mask: Máscara para ocultar elementos de relleno en tgt_a.
    #   tgt_b_key_padding_mask: Máscara para ocultar elementos de relleno en tgt_b.
    #   memory_key_padding_mask: Máscara para ocultar elementos de relleno en la memoria.
  
    def forward(self, src, tgt_a, tgt_b, src_mask, tgt_a_mask, tgt_b_mask,  memory_mask, src_key_padding_mask, tgt_a_key_padding_mask, tgt_b_key_padding_mask, memory_key_padding_mask):
        # Verifica si los datos de entrada tienen la forma correcta y el tamaño adecuado.
        is_batched = src.dim() == 3
        if src.size(1) != tgt_a.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt_a must be equal")
        if src.size(-1) != self.d_model or tgt_a.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt_a must be equal to d_model")
        # Pasa los datos de entrada a través del codificador y almacena la salida en memory.
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        # Luego, pasa los datos de entrada de cada decodificador junto con memory a través de los decodificadores correspondientes.
        logits_for_a = self.decoder_a(tgt_a, memory, tgt_a_mask, memory_mask, tgt_a_key_padding_mask, memory_key_padding_mask)
        logits_for_b = self.decoder_b(tgt_b, memory, tgt_b_mask, memory_mask, tgt_b_key_padding_mask, memory_key_padding_mask)
        # La salida final de ambos decodificadores se devuelve como resultado.
        return logits_for_a, logits_for_b
    
# La clase CustomSeq2SeqTransformer representa un modelo de Transformer completo para la traducción de secuencia a secuencia (Seq2Seq). 

class CustomSeq2SeqTransformer(nn.Module):

    # Esta función se llama cuando se instancia un objeto de la clase CustomSeq2SeqTransformer.
    # Los parámetros de entrada son:
    #   name: Nombre del modelo.
    #   num_encoder_layers: Número de capas en el codificador.
    #   num_decoder_layers: Número de capas en el decodificador.
    #   emb_size: Dimensión de los vectores de embedding.
    #   head_num: Número de cabezas de atención en las capas de atención.
    #   src_vocab_size: Tamaño del vocabulario de la secuencia fuente.
    #   tgt_vocab_size: Tamaño del vocabulario de la secuencia objetivo.
    #   ff_dim: Dimensión de la capa de feedforward.
    #   dropout: Tasa de dropout para las capas de dropout.

    def __init__(self, name, num_encoder_layers, num_decoder_layers, emb_size, head_num, src_vocab_size,  tgt_vocab_size, ff_dim = 512, dropout: float = 0.1):
        super(CustomSeq2SeqTransformer, self).__init__()
        self.name = name
        # Un modelo de Transformer personalizado con un solo decodificador.
        self.transformer = CustomSingleDecoderTransformer(d_model=emb_size, head_num=head_num, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, ff_dim=ff_dim, dropout=dropout)
        # Una capa lineal para generar salidas a partir de los vectores de embedding.
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # Un módulo de embedding para tokens de la secuencia fuente.
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # Un módulo de embedding para tokens de la secuencia objetivo.
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # Un módulo de codificación posicional para agregar información de posición a los vectores de embedding.
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
    
    # Este método define cómo se calcula la salida del modelo durante el entrenamiento.
    
    # Los parámetros de entrada son:
    #   src: Secuencia de entrada.
    #   trg: Secuencia objetivo.
    #   src_mask: Máscara para ocultar tokens en la secuencia fuente.
    #   tgt_mask: Máscara para ocultar tokens futuros en la secuencia objetivo.
    #   src_padding_mask: Máscara para ocultar elementos de relleno en la secuencia fuente.
    #   tgt_padding_mask: Máscara para ocultar elementos de relleno en la secuencia objetivo.
    #   memory_key_padding_mask: Máscara para ocultar elementos de relleno en la memoria.

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        # Convierte las secuencias de entrada y objetivo en vectores de embedding, agregando información de posición.
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # Utiliza el modelo Transformer para calcular la salida.
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # Aplica una capa lineal para generar la salida final.
        return self.generator(outs)

    def encode(self, src, src_mask):
        # Convierte la secuencia de entrada en vectores de embedding y calcula la representación codificada utilizando el codificador del modelo Transformer.
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        # Convierte la secuencia de entrada en vectores de embedding y calcula la salida utilizando el decodificador del modelo Transformer.
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)    
    
    def getName(self):
        # Este método devuelve el nombre del modelo.
        return self.name   




# Seq2Seq Network
# La clase Seq2SeqTransformer es un modelo de Transformer diseñado específicamente para tareas de traducción de secuencia a secuencia (Seq2Seq). 

class Seq2SeqTransformer(nn.Module):
    """
    Esta función se llama cuando se instancia un objeto de la clase Seq2SeqTransformer.
    Los parámetros de entrada son:
        name: Nombre del modelo.
        num_encoder_layers: Número de capas en el codificador.
        num_decoder_layers: Número de capas en el decodificador.
        emb_size: Dimensión de los vectores de embedding.
        nhead: Número de cabezas de atención en las capas de atención.
        src_vocab_size: Tamaño del vocabulario de la secuencia fuente.
        tgt_vocab_size: Tamaño del vocabulario de la secuencia objetivo.
        ff_dim: Dimensión de la capa de feedforward.
        dropout: Tasa de dropout para las capas de dropout.
    """
    def __init__(self, name, num_encoder_layers,  num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size,  ff_dim = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.name = name
        # Un modelo de Transformer estándar con un codificador y un decodificador.
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=ff_dim, dropout=dropout)
        #  Una capa lineal para generar salidas a partir de los vectores de embedding.
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # Un módulo de embedding para tokens de la secuencia fuente.
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # Un módulo de embedding para tokens de la secuencia objetivo.
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # Un módulo de codificación posicional para agregar información de posición a los vectores de embedding.
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    """
    Este método define cómo se calcula la salida del modelo durante el entrenamiento.
    Los parámetros de entrada son similares a los del constructor, pero también incluyen máscaras para ocultar tokens y elementos de relleno.
    """

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        # Convierte las secuencias de entrada y objetivo en vectores de embedding, agregando información de posición.
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # Utiliza el modelo Transformer para calcular la salida.
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # Aplica una capa lineal para generar la salida final.
        return self.generator(outs)

    # Este método permite realizar únicamente la codificación de una secuencia de entrada.
    def encode(self, src, src_mask):
        # Convierte la secuencia de entrada en vectores de embedding y calcula la representación codificada 
        # utilizando el codificador del modelo Transformer.
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    # Este método permite realizar únicamente la decodificación utilizando la memoria codificada y una secuencia 
    # de entrada en el decodificador.
    def decode(self, tgt, memory, tgt_mask):
        # Convierte la secuencia de entrada en vectores de embedding y calcula la salida utilizando el 
        # decodificador del modelo Transformer.
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask) 

    def getName(self):
        # Este método devuelve el nombre del modelo.
        return self.name   




# ----------------------------------------------------------------------------------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------------------------------------------------------------------------------
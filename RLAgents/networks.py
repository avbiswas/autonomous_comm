import tensorflow as tf


def single_head_attention(query, key, scope_name="attention", reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope_name, reuse=reuse):
        query_embedding = tf.keras.layers.Dense(64)(query)
        key_embedding = tf.keras.layers.Dense(64)(key)
        value_embedding = tf.keras.layers.Dense(64, activation=tf.nn.relu)(key)
        added = query_embedding + key_embedding
        attention_scores = tf.keras.layers.Dense(1, activation=tf.nn.tanh)(added)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        context_vector = tf.reduce_sum(attention_scores * value_embedding, axis=1)
    return attention_scores, context_vector


def multiheaded_attention(query, key, n_heads, units,
                          scope_name="multiattention", reuse=tf.AUTO_REUSE):
    emb_size = units//n_heads
    # units = emb_size * n_heads
    with tf.variable_scope(scope_name, reuse=reuse):
        print(tf.shape(key))
        print()
        print(query, key)
        query_embedding = tf.keras.layers.Dense(units)(query)
        key_embedding = tf.keras.layers.Dense(units)(key)
        value_embedding = tf.keras.layers.Dense(units, activation=tf.nn.relu)(key)
        print(query_embedding, key_embedding)

        query_reshaped = tf.reshape(query_embedding, (-1, 1, n_heads, emb_size))
        key_reshaped = tf.reshape(key_embedding, (-1, tf.shape(key)[1], n_heads, emb_size))
        value_reshaped = tf.reshape(value_embedding, (-1, tf.shape(key)[1], n_heads, emb_size))
        print(query_reshaped, key_reshaped)

        # exit()

        query_reshaped = tf.transpose(query_reshaped, [0, 2, 1, 3])
        key_reshaped = tf.transpose(key_reshaped, [0, 2, 1, 3])
        value_reshaped = tf.transpose(value_reshaped, [0, 2, 1, 3])
        print(query_reshaped, key_reshaped, value_reshaped)

        query_embedding2 = tf.keras.layers.Dense(emb_size)(query_reshaped)
        key_embedding2 = tf.keras.layers.Dense(emb_size)(key_reshaped)
        value_embedding2 = tf.keras.layers.Dense(emb_size)(value_reshaped)
        print(query_embedding2, key_embedding2, value_embedding2)

        added = query_embedding2 + key_embedding2
        print(added)
        multi_attention_scores = tf.keras.layers.Dense(1)(added)
        multi_attention_scores = tf.nn.softmax(multi_attention_scores, axis=-2)
        print(multi_attention_scores)
        contexts = tf.reduce_sum(multi_attention_scores * value_embedding2, axis=-2)
        print(contexts)
        context_vector = tf.reshape(contexts, [-1, units])
        print(context_vector)
    return multi_attention_scores, context_vector


def NeighborhoodEncoder(input):
    with tf.variable_scope("Nenc", reuse=tf.AUTO_REUSE):
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same',
                                      data_format='channels_first', activation=tf.nn.relu)(input)
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same',
                                      data_format='channels_first', activation=tf.nn.relu)(conv)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same',
                                      data_format='channels_first', activation=tf.nn.relu)(conv)

        conv = tf.keras.layers.Flatten()(conv)
        return None, conv


def ImageEncoder(input):
    with tf.variable_scope("Conv", reuse=tf.AUTO_REUSE):
        print(input)
        init = tf.random_normal_initializer(0, 0.02)
        input = input/127.5 - 1
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=4,
                                      data_format='channels_first', kernel_initializer=init,
                                      activation=tf.nn.relu)(input)
        print(conv)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                      data_format='channels_first', kernel_initializer=init,
                                      activation=tf.nn.relu)(conv)
        print(conv)
        # conv = tf.keras.layers.A(data_format='channels_first')(conv)
        print(conv)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                      data_format='channels_first', kernel_initializer=init,
                                      activation=tf.nn.relu)(conv)
        print(conv)
        conv = tf.keras.layers.Flatten()(conv)
        conv = tf.keras.layers.Dense(512, activation=tf.nn.relu)(conv)
        print(conv)
        return None, conv


def AttentionKinematicsEncoder(input, heads=8):
    def get_neighbor_encodings(neighbor_feats, neighbor_masks, reuse=False):
        with tf.variable_scope("encoding", reuse=tf.AUTO_REUSE):
            dense = tf.keras.layers.Dense(64, activation=tf.nn.tanh)(neighbor_feats)
            # dense = tf.keras.layers.Dense(32, activation=tf.nn.tanh)(dense)
            dense = dense * neighbor_masks
        return dense

    def get_agent_encoding(agent_feats):
        with tf.variable_scope("agent_encoding", reuse=tf.AUTO_REUSE):
            dense = tf.keras.layers.Dense(64, activation=tf.nn.tanh)(agent_feats)
        # dense = tf.keras.layers.Dense(32, activation=tf.nn.tanh)(dense)
        return dense

    def get_attention_scores(neighbor_encodings, agent_encodings, reuse=False):
        # with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
        query_ = tf.reduce_sum(neighbor_encodings, axis=1, keepdims=True)
        query_2 = tf.reduce_max(neighbor_encodings, axis=1, keepdims=True)
        query_concat = tf.concat([query_, query_2, agent_encodings], axis=-1)
        if heads == 1:
            return single_head_attention(query=query_concat,
                                         key=neighbor_encodings)
        else:
            return multiheaded_attention(query=query_concat,
                                         key=neighbor_encodings,
                                         n_heads=heads,
                                         units=heads*32)

    neighbor_feats = input[:, 1:, 1:]
    neighbor_masks = input[:, 1:, 0:1]
    agent_feats = input[:, 0:1, 1:]
    neighbor_encodings = get_neighbor_encodings(neighbor_feats,
                                                neighbor_masks)
    agent_encoding = get_agent_encoding(agent_feats)
    return get_attention_scores(neighbor_encodings, agent_encoding)

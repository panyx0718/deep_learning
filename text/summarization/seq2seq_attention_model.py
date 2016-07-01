"""Automatic text summarization using seq2seq model with attention.
"""

# Import files removed.

# tensor names for exported model.
# Note: Please keep in sync with the exported model serving code.
SIGNATURE_NAME = 'generic'
ARTICLE = 'article'
ARTICLE_LEN = 'article_len'
ABSTRACTS = 'abstract'
OUTPUT = 'output'


def default_hparams():
  return tf.HParams(mode='train',  # train, eval, decode
                    min_lr=0.01,  # min learning rate.
                    lr=0.15,  # learning rate
                    batch_size=64,
                    enc_layers=4,
                    enc_timesteps=120,
                    dec_timesteps=30,
                    num_hidden=256,  # for rnn cell
                    emb_dim=128,  # If 0, don't use embedding
                    num_heads=1,
                    max_grad_norm=2,
                    num_softmax_samples=4096,  # If 0, don't use sample softmax.
                    dropout_rnn=1.0,
                    dropout_emb=1.0,
                    # pyramid structure: http://arxiv.org/abs/1508.01211
                    # Not helpful, hence not enabled.
                    num_pyramids=0)


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    """function that feed previous model output rather than ground truth."""
    if output_projection is not None:
      prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = tf.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


class Seq2SeqAttentionModel(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, hps, vocab, num_gpus=0):
    self._hps = hps
    self._vocab = vocab
    self._num_gpus = num_gpus
    self._cur_gpu = 0

  def run_train_step(self, sess, article_batch, abstract_batch, targets,
                     article_lens, abstract_lens, loss_weights):
    to_return = [self._train_op, self._summaries, self._loss, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def run_eval_step(self, sess, article_batch, abstract_batch, targets,
                    article_lens, abstract_lens, loss_weights):
    to_return = [self._summaries, self._loss, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def run_decode_step(self, sess, article_batch, abstract_batch, targets,
                      article_lens, abstract_lens, loss_weights):
    to_return = [self._outputs, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def _next_device(self):
    """Round robin the gpu device. (Reserve last gpu for expensive op)."""
    if self._num_gpus == 0:
      return ''
    dev = '/gpu:%d' % self._cur_gpu
    self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus-1)
    return dev

  def _get_gpu(self, gpu_id):
    if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
      return ''
    return '/gpu:%d' % gpu_id

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self._hps
    self._articles = tf.placeholder(tf.int32,
                                    [hps.batch_size, hps.enc_timesteps],
                                    name='articles')
    self._abstracts = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps],
                                     name='abstracts')
    self._targets = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='targets')
    self._article_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='article_lens')
    self._abstract_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                         name='abstract_lens')
    self._loss_weights = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                        name='loss_weights')

  def _add_seq2seq(self):
    hps = self._hps
    vsize = self._vocab.NumIds()

    with tf.variable_scope('seq2seq'):
      encoder_inputs = tf.unpack(self._articles, axis=1)
      decoder_inputs = tf.unpack(self._abstracts, axis=1)
      targets = tf.unpack(self._targets, axis=1)
      loss_weights = tf.unpack(self._loss_weights, axis=1)
      article_lens = self._article_lens

      # Embedding shared by the input and outputs.
      with tf.variable_scope('embedding'), tf.device('/cpu:0'):
        embedding = tf.get_variable(
            'embedding', [vsize, hps.emb_dim], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in encoder_inputs]
        emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in decoder_inputs]

      for layer_i in xrange(hps.enc_layers):
        with tf.variable_scope('encoder%d'%layer_i), tf.device(
            self._next_device()):
          cell_fw = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
          cell_bw = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113))
          (emb_encoder_inputs, fw_state) = seq2seq_lib.bidirectional_rnn(
              cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32,
              sequence_length=article_lens)
          # Experimental: Pyramid structure to reduce input sequence length.
          if layer_i < hps.num_pyramids:
            red_enc_inputs = []
            for enc_i in xrange(0, len(emb_encoder_inputs), 2):
              red_enc = tf.concat(
                  1,
                  [emb_encoder_inputs[enc_i],
                   emb_encoder_inputs[min(len(emb_encoder_inputs)-1, enc_i+1)]])
              red_enc_inputs.append(red_enc)
              article_lens /= 2
            emb_encoder_inputs = red_enc_inputs
      encoder_outputs = emb_encoder_inputs

      with tf.variable_scope('output_projection'):
        w = tf.get_variable(
            'w', [hps.num_hidden, vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        w_t = tf.transpose(w)
        v = tf.get_variable(
            'v', [vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

      with tf.variable_scope('decoder'), tf.device(self._next_device()):
        # When decoding, use model output from the previous step
        # for the next step.
        loop_function = None
        if hps.mode == 'decode' or hps.mode == 'export':
          loop_function = _extract_argmax_and_embed(
              embedding, (w, v), update_embedding=False)

        cell = tf.nn.rnn_cell.LSTMCell(
            hps.num_hidden,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113))

        encoder_outputs = [tf.reshape(x, [hps.batch_size, 1, 2*hps.num_hidden])
                           for x in encoder_outputs]
        self._enc_top_states = tf.concat(1, encoder_outputs)
        self._dec_in_state = fw_state
        # During decoding, follow up _dec_in_state are fed from beam_search.
        # dec_out_state are stored by beam_search for next step feeding.
        initial_state_attention = (hps.mode == 'decode' or hps.mode == 'export')
        decoder_outputs, self._dec_out_state = tf.nn.seq2seq.attention_decoder(
            emb_decoder_inputs, self._dec_in_state, self._enc_top_states,
            cell, num_heads=hps.num_heads, loop_function=loop_function,
            initial_state_attention=initial_state_attention)

      with tf.variable_scope('output'), tf.device(self._next_device()):
        model_outputs = []
        for i in xrange(len(decoder_outputs)):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          model_outputs.append(
              tf.nn.xw_plus_b(decoder_outputs[i], w, v))

      if hps.mode == 'decode' or hps.mode == 'export':
        with tf.variable_scope('decode_output'), tf.device('/cpu:0'):
          best_outputs = [tf.argmax(x, 1) for x in model_outputs]
          tf.logging.info('best_outputs%s', best_outputs[0].shape)
          self._outputs = tf.concat(
              1, [tf.reshape(x, [hps.batch_size, 1]) for x in best_outputs])

          self._topk_log_probs, self._topk_ids = tf.nn.top_k(
              tf.log(tf.nn.softmax(model_outputs[-1])), hps.batch_size*2)

      with tf.variable_scope('loss'), tf.device(self._next_device()):
        def sampled_loss_func(inputs, labels):
          with tf.device('/cpu:0'):  # Try gpu.
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(w_t, v, inputs, labels,
                                              hps.num_softmax_samples, vsize)

        if hps.num_softmax_samples != 0 and hps.mode == 'train':
          self._loss = seq2seq_lib.sampled_sequence_loss(
              decoder_outputs, targets, loss_weights, sampled_loss_func)
        else:
          self._loss = tf.nn.seq2seq.sequence_loss(
              model_outputs, targets, loss_weights)
        tf.scalar_summary('loss', tf.minimum(12.0, self._loss))

  def _add_train_op(self):
    """Sets self._train_op, op to run for training."""
    hps = self._hps

    self._lr_rate = tf.maximum(
        hps.min_lr,  # min_lr_rate.
        tf.train.exponential_decay(hps.lr, self.global_step, 30000, 0.98))

    tvars = tf.trainable_variables()
    with tf.device(self._get_gpu(self._num_gpus-1)):
      grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self._loss, tvars), hps.max_grad_norm)
    tf.scalar_summary('global_norm', global_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
    tf.scalar_summary('learning rate', self._lr_rate)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step, name='train_step')

  def encode_top_state(self, sess, enc_inputs, enc_len):
    """Return the top states from encoder for decoder.

    Args:
      sess: tensorflow session.
      enc_inputs: encoder inputs of shape [batch_size, enc_timesteps].
      enc_len: encoder input length of shape [batch_size]
    Returns:
      enc_top_states: The top level encoder states.
      dec_in_state: The decoder layer initial state.
    """
    results = sess.run([self._enc_top_states, self._dec_in_state],
                       feed_dict={self._articles: enc_inputs,
                                  self._article_lens: enc_len})
    return results[0], results[1][0]

  def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states):
    """Return the topK results and new decoder states."""
    feed = {
        self._enc_top_states: enc_top_states,
        self._dec_in_state:
            np.squeeze(np.array(dec_init_states)),
        self._abstracts:
            np.transpose(np.array([latest_tokens])),
        self._abstract_lens: np.ones([len(dec_init_states)], np.int32)}

    results = sess.run(
        [self._topk_ids, self._topk_log_probs, self._dec_out_state],
        feed_dict=feed)

    ids, probs, states = results[0], results[1], results[2]
    new_states = [s for s in states]
    return ids, probs, new_states

  def build_graph(self):
    self._add_placeholders()
    self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.merge_all_summaries()

  def get_signature(self):
    """Return the signature for model export."""
    # Abstract is no needed for inference. Here, the abstract
    # is used to kick off the decoding. So "<s> <s> ..." input should be good.
    named_tensor_bindings = {ARTICLE: self._articles,
                             ARTICLE_LEN: self._article_lens,
                             ABSTRACTS: self._abstracts,
                             OUTPUT: self._outputs}
    signatures = {SIGNATURE_NAME:
                  exporter.generic_signature(named_tensor_bindings)}
    return signatures

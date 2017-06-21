# coding: utf-8

# 

# In[1]:


FN = 'train'


# In[2]:


import os
import tensorflow as tf
# os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'


# In[3]:


import keras
#tf.__version__
keras.__version__


# In[4]:


FN0 = 'vocabulary-embedding'


# In[5]:


FN1 = 'train'


# In[6]:


maxlend=25 # 0 - if we dont want to use description at all
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512 # must be same as 160330-word-gen
rnn_layers = 3  # match FN1
batch_norm=False


# In[7]:


activation_rnn_size = 40 if maxlend else 0


# In[8]:


# training parameters
seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size=32
nflips=10


# In[9]:


nb_train_samples = 30000 #60000#0
nb_val_samples = 3000 #10000#0


# ### Read word embedding

# In[10]:


import pickle as pickle

with open('data/%s.pkl'%FN0, 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape
print(vocab_size, embedding_size)


# In[11]:


with open('data/%s.data.pkl'%FN0, 'rb') as fp:
    X, Y = pickle.load(fp)


# In[12]:


nb_unknown_words = 10


# In[13]:


print ('number of examples',len(X),len(Y))
print ('dimension of embedding space for words',embedding_size)
print ('vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words'%nb_unknown_words)
print ('total number of different words',len(idx2word), len(word2idx))
print ('number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx))
print ('number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(glove_idx2idx))


# In[14]:


for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i


# In[15]:


oov0 = vocab_size-nb_unknown_words


# In[16]:


for i in range(oov0, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'


# In[17]:


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)

X_train = X_train[:30000]
Y_train = Y_train[:30000]
len(X_train), len(Y_train), len(X_test), len(Y_test)


# In[18]:


del X
del Y


# In[19]:


empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'


# In[20]:


import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys


# In[21]:


def prt(label, x):
    print(label+':', end=' ')
    for w in x:
        print(idx2word[w] , end=' ')
    print ()


# In[22]:


i = 334
prt('H',Y_train[i])
prt('D',X_train[i])


# In[23]:


i = 334
prt('H',Y_test[i])
prt('D',X_test[i])


# ## Model

# In[24]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers import merge, SpatialDropout1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2


# In[25]:


# seed weight initialization
random.seed(seed)
np.random.seed(seed)


# In[26]:


regularizer = l2(weight_decay) if weight_decay else None


# In[27]:


model = Sequential()
#keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', 
#            embeddings_regularizer=None, activity_regularizer=None, 
#            embeddings_constraint=None, mask_zero=False, input_length=None)

model.add(Embedding(input_dim=vocab_size, 
                    output_dim=embedding_size,
                    embeddings_initializer=keras.initializers.Zeros(),#'uniform',
                    embeddings_regularizer=regularizer,
                    mask_zero=True,
                    input_length=maxlen,
                    name='embedding_1'))

'''model.add(Embedding(vocab_size, 
                    embedding_size,
                    input_length=maxlen,
                    W_regularizer=regularizer,
                    dropout=p_emb, weights=[embedding],
                    mask_zero=True,
                    name='embedding_1'))'''

model.add(SpatialDropout1D(p_emb, name='dropout_emb_1'))

for i in range(rnn_layers):
    #New
    #keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', 
    #     use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
    #     bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, 
    #     bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, 
    #     bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
    
    lstm = LSTM(units=rnn_size,
                return_sequences=True,
                kernel_regularizer=regularizer, #kernel_regularizer
                recurrent_regularizer=regularizer, #recurrent_regularizer
                bias_regularizer=regularizer, #bias_regularizer
                dropout=p_W, #dropout
                recurrent_dropout=p_U, #recurrent_dropout
                name='lstm_%d'%(i+1)
                  )
    '''lstm = LSTM(rnn_size,
                return_sequences=True,
                W_regularizer=regularizer,
                U_regularizer=regularizer, 
                b_regularizer=regularizer,
                dropout_W=p_W, 
                dropout_U=p_U,
                name='lstm_%d'%(i+1)
                  )'''
    model.add(lstm)
    model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))


# In[28]:


from keras.layers.core import Lambda
import keras.backend as K

def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlenh:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]

    #print(maxlend)
    #print(maxlenh)
    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    if mask is not None:
        activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
    
    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))
    
    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    #print(desc_words)
    return K.concatenate((desc_avg_word, head_words))
        

class SimpleContext(Lambda):
    def __init__(self,**kwargs):
        super(SimpleContext, self).__init__(simple_context,**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]
    
    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)


# In[29]:


if activation_rnn_size:
    model.add(SimpleContext(name='simplecontext_1'))

#keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
#     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
#     kernel_constraint=None, bias_constraint=None)
model.add(TimeDistributed(Dense(units=vocab_size,
                                kernel_regularizer=regularizer, 
                                bias_regularizer=regularizer,
                                name = 'timedistributed_1')))

'''model.add(TimeDistributed(Dense(vocab_size,
                                W_regularizer=regularizer, 
                                b_regularizer=regularizer,
                                name = 'timedistributed_1')))'''

model.add(Activation('softmax', name='activation_1'))


# In[30]:


from keras.optimizers import Adam, RMSprop # usually I prefer Adam but article used rmsprop
# opt = Adam(lr=LR)  # keep calm and reduce learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[31]:


#get_ipython().run_cell_magic(u'javascript', u'', u'// new Audio("http://www.soundjay.com/button/beep-09.wav").play ()')


# In[32]:


K.set_value(model.optimizer.lr,np.float32(LR))


# In[33]:


def str_shape(x):
    print("--- ", x.shape)
    return 'x'.join(map(str,x.shape))
    
def inspect_model(model):
    for i,l in enumerate(model.layers):
        print (i, 'cls=%s name=%s'%(type(l).__name__, l.name))
        weights = l.get_weights()
        print("*** ", len(weights))
        for weight in weights:
            print(str_shape(weight), end=' ')
        print()


# In[34]:


inspect_model(model)


# ## Load

# In[35]:


#if FN1:
#    model.load_weights('data/%s.hdf5'%FN1)


# ## Test

# In[36]:


def lpadd(x, maxlend=maxlend, eos=eos):
    """left (pre) pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
        
    #print([empty]*(maxlend-n) + x + [eos])
    return [empty]*(maxlend-n) + x + [eos]


# In[37]:


samples = [lpadd([3]*26)]
# pad from right (post) so the first maxlend will be description followed by headline
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')


# In[38]:


np.all(data[:,maxlend] == eos)


# In[39]:


data.shape,list(map(len, samples))


# In[40]:


probs = model.predict(data, verbose=0, batch_size=1)
probs.shape


# ## Sample Generation

# In[41]:


# variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
def beamsearch(predict, start=[empty]*maxlend + [eos],
               k=1, maxsample=maxlen, use_unk=True, empty=empty, eos=eos, temperature=1.0):
    """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
    all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
    """
    def sample(energy, n, temperature=temperature):
        """sample at most n elements according to their energy"""
        n = min(n,len(energy))
        prb = np.exp(-np.array(energy) / temperature )
        res = []
        for i in range(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb/z, 1))
            res.append(r)
            prb[r] = 0. # make sure we select each element only once
        return res

    dead_k = 0 # samples that reached eos
    dead_samples = []
    dead_scores = []
    live_k = 1 # samples that did not yet reached eos
    live_samples = [list(start)]
    live_scores = [0]

    while live_k:
        # for every possible live sample calc prob for every possible label 
        probs = predict(live_samples, empty=empty)

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:,None] - np.log(probs)
        cand_scores[:,empty] = 1e20
        if not use_unk:
            for i in range(nb_unknown_words):
                cand_scores[:,vocab_size - 1 - i] = 1e20
        live_scores = list(cand_scores.flatten())
        

        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)

        ranks_dead = [r for r in ranks if r < n]
        ranks_live = [r - n for r in ranks if r >= n]
        
        dead_scores = [dead_scores[r] for r in ranks_dead]
        dead_samples = [dead_samples[r] for r in ranks_dead]
        
        live_scores = [live_scores[r] for r in ranks_live]

        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        live_samples = [live_samples[r//voc_size]+[r%voc_size] for r in ranks_live]

        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of maxlenh
        zombie = [s[-1] == eos or len(s) > maxsample for s in live_samples]
        
        # add zombies to the dead
        dead_samples += [s for s,z in zip(live_samples,zombie) if z]
        dead_scores += [s for s,z in zip(live_scores,zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living 
        live_samples = [s for s,z in zip(live_samples,zombie) if not z]
        live_scores = [s for s,z in zip(live_scores,zombie) if not z]
        live_k = len(live_samples)

    return dead_samples + live_samples, dead_scores + live_scores


# In[42]:


def keras_rnn_predict(samples, empty=empty, model=model, maxlen=maxlen):
    """for every sample, calculate probability for every possible label
    you need to supply your RNN model and maxlen - the length of sequences it can handle
    """
    sample_lengths = list(map(len, samples))
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    # pad from right (post) so the first maxlend will be description followed by headline
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    return np.array([prob[sample_length-maxlend-1] for prob, sample_length in zip(probs, sample_lengths)])


# In[43]:


def vocab_fold(xs):
    """convert list of word indexes that may contain words outside vocab_size to words inside.
    If a word is outside, try first to use glove_idx2idx to find a similar word inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    xs = [x if x < oov0 else glove_idx2idx.get(x,x) for x in xs]
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= oov0])
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs


# In[44]:


def vocab_unfold(desc,xs):
    # assume desc is the unfolded version of the start of xs
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= oov0:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x,x) for x in xs]


# In[45]:


import sys
import Levenshtein

def gensamples(skips=2, k=10, batch_size=batch_size, short=True, temperature=1., use_unk=True):
    i = random.randint(0,len(X_test)-1)
    print ('HEAD:',' '.join(idx2word[w] for w in Y_test[i][:maxlenh]))
    print ('DESC:',' '.join(idx2word[w] for w in X_test[i][:maxlend]))
    sys.stdout.flush()

    print ('HEADS:')
    x = X_test[i]
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start)
        sample, score = beamsearch(predict=keras_rnn_predict, start=fold_start, k=k, temperature=temperature, use_unk=use_unk)
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s,start,scr) for s,scr in zip(sample,score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
            if distance > -0.6:
                print (score, ' '.join(words))
                #print ('%s (%.2f) %f'%(' '.join(words), score, distance))
        else:
                print (score, ' '.join(words))
        codes.append(code)


# In[46]:


gensamples(skips=2, batch_size=batch_size, k=10, temperature=1.)


# ## Data Generator

# In[47]:


def flip_headline(x, nflips=None, model=None, debug=False):
    """given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
    with words predicted by the model
    """
    if nflips is None or model is None or nflips <= 0:
        return x
    
    batch_size = len(x)
    assert np.all(x[:,maxlend] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
        flips = sorted(random.sample(range(maxlend+1,maxlen), nflips))
        if debug and b < debug:
            print(b)
        for input_idx in flips:
            if x[b,input_idx] == empty or x[b,input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlend+1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = oov0
            if debug and b < debug:
                print ('%s => %s'%(idx2word[x_out[b,input_idx]],idx2word[w]))
            x_out[b,input_idx] = w
        #if debug and b < debug:
        #    print
    return x_out


# In[48]:


def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [vocab_fold(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug)
    
    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh) + [eos] + [empty]*maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i,:,:] = np_utils.to_categorical(xh, vocab_size)
        
    return x, y


# In[49]:


def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
    """yield batches. for training use nb_batches=None
    for validation generate deterministic results repeating every nb_batches
    
    while training it is good idea to flip once in a while the values of the headlines from the
    value taken from Xh to value generated by the model.
    """
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, sys.maxsize)
        random.seed(c+123456789+seed)
        for b in range(batch_size):
            t = random.randint(0,len(Xd)-1)

            xd = Xd[t]
            s = random.randint(min(maxlend,len(xd)), max(maxlend,len(xd)))
            xds.append(xd[:s])
            
            xh = Xh[t]
            s = random.randint(min(maxlenh,len(xh)), max(maxlenh,len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c+= 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)


# In[50]:


r = next(gen(X_train, Y_train, batch_size=batch_size))
r[0].shape, r[1].shape, len(r)


# In[51]:


def test_gen(gen, n=5):
    Xtr,Ytr = next(gen)
    for i in range(n):
        assert Xtr[i,maxlend] == eos
        x = Xtr[i,:maxlend]
        y = Xtr[i,maxlend:]
        yy = Ytr[i,:]
        yy = np.where(yy)[1]
        prt('L',yy)
        prt('H',y)
        if maxlend:
            prt('D',x)


# In[52]:


test_gen(gen(X_train, Y_train, batch_size=batch_size))


# ### Test Flipping

# In[53]:


test_gen(gen(X_train, Y_train, nflips=6, model=model, debug=False, batch_size=batch_size))


# In[54]:


valgen = gen(X_test, Y_test,nb_batches=3, batch_size=batch_size)


# Check that valgen repeats itself after nb_batches

# In[55]:


for i in range(4):
    test_gen(valgen, n=1)


# ## Train

# In[56]:


history = {}


# In[57]:


#model


# In[58]:


traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
valgen = gen(X_test, Y_test, nb_batches=nb_val_samples//batch_size, batch_size=batch_size)


# In[ ]:


for iteration in range(500):
    print ('Iteration', iteration)
    #fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, 
    #     validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, 
    #     workers=1, pickle_safe=False, initial_epoch=0)
    h = model.fit_generator(generator=traingen, 
                            steps_per_epoch=nb_train_samples,
                            epochs=1, 
                            validation_data=valgen, 
                            validation_steps=nb_val_samples,
                            initial_epoch=0
                           )
    for k,v in h.history.items(): #iteritems():
        history[k] = history.get(k,[]) + v
    with open('data/%s.history.pkl'%FN,'wb') as fp:
        pickle.dump(history,fp,-1)
    model.save_weights('data/%s.hdf5'%FN, overwrite=True)
    gensamples(batch_size=batch_size)


# In[ ]:
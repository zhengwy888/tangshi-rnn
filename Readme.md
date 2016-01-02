A fork of [char-rnn](https://github.com/karpathy/char-rnn) that learns from line samples, excellent at producing structured poems.  
Demo at [http://zhengwy.com/make-tangshi/](http://zhengwy.com/make-tangshi/)
See [my blog](zhengwy.com/neural-network-for-tangshi/) that explains some details.

## Train
To train this is the easiest (the following command is what worked for me, smaller network will not work so well)
```
th lstm_rnn.lua -data_dir data/qts_jinti_dummy -batch_size 20 -max_epochs 25 -eval_val_every 0.5 -seq_length 25 -rnn_size 500 -print_every 10 -savemodel 1 -train_frac 0.9 -val_frac 0.1 -learning_rate_decay_after 5 -dropout 0.2
```
if you are running into out of memory issue, reduce the rnn_size or batch_size

The models are saved along the way. Pick the one with the lowest error, not the last one. 

## Data
I only included a small sample of Quan-Tang-Shi. If you want the full dataset please google and retrieve it on your own.

## Sample
To sample you have two options
To use the sampling function from the original char-rnn
```
th sample.lua <model.t7>
```
This will produce a bunch of text, 
To Sample with Pingze rules and end-of-poem characters, use sample_qts.lua
```
th sample_qts.lua <model.t7>
```

Be warned, the Yunjiao module is not complete in terms of dictionary, so if you see this error
```
./yunjiao/YunjiaoLoader.lua:150: out of range char 祚
```
That means you have to go to some Pingshuiyun site and find that character, and add it to the correct line in either ping.txt or ze.txt. One possible such site is [wikisource](https://zh.wikisource.org/zh/%E5%B9%B3%E6%B0%B4%E9%9F%BB)

To see what the parameter means, read the original documentation at [char-rnn](https://github.com/karpathy/char-rnn)

##License
MIT


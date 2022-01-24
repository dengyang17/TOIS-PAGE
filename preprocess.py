import argparse
import pickle

import utils

parser = argparse.ArgumentParser(description='preprocess.py')

parser.add_argument('--load_data', required=True,
                    help="input file for the data")

parser.add_argument('--save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('--src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('--tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('--bow_vocab_size', type=int, default=10000,
                    help="Size of the bow vocabulary")
parser.add_argument('--stopwords', type=str, default=None,
                    help="stopwords file")

parser.add_argument('--src_filter', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('--tgt_filter', type=int, default=0,
                    help="Maximum target sequence length")
parser.add_argument('--src_trun', type=int, default=0,
                    help="Truncate source sequence length")
parser.add_argument('--tgt_trun', type=int, default=0,
                    help="Truncate target sequence length")
parser.add_argument('--src_char', action='store_true',
                    help='character based encoding')
parser.add_argument('-tgt_char', action='store_true',
                    help='character based decoding')
parser.add_argument('--src_dict', help='')
parser.add_argument('--tgt_dict', help='')
parser.add_argument('--src_suf', default='src',
                    help="the suffix of the source filename")
parser.add_argument('--tgt_suf', default='tgt',
                    help="the suffix of the target filename")
parser.add_argument('--fact_suf', default='fact',
                    help="the suffix of the fact filename")
parser.add_argument('--pers_suf', default='pers',
                    help="the suffix of the persona filename")
parser.add_argument('--lower', action='store_false',
                    help='lower the case')
parser.add_argument('--share', action='store_false',
                    help='share the vocabulary between source and target')
parser.add_argument('--freq', type=int, default=0,
                    help="remove words less frequent")

parser.add_argument('--report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(filename, trun_length, filter_length, char, vocab, size, bow_size=0, stopwords=None, freq=0):

    print("%s: length limit = %d, truncate length = %d" %
          (filename, filter_length, trun_length))
    max_length = 0
    with open(filename, encoding='utf8') as f:
        for sent in f:
            try:
                sent = ' '.join(eval(sent))
            except Exception as e:
                pass
            if char:
                tokens = list(sent.strip())
            else:
                tokens = sent.strip().split()
            if 0 < filter_length < len(sent.strip().split()):
                continue
            max_length = max(max_length, len(tokens))
            if trun_length > 0:
                tokens = tokens[:trun_length]
            for word in tokens:
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))

    

    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size, freq)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))

    if bow_size > 0:
       actual_bow_vocab = vocab.init_bow_vocab(bow_size, stopwords, freq)
       print('Created bow dictionary of size %d (expected %d)' %
              (actual_bow_vocab, bow_size)) 
    
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, factFile, persFile, srcDicts, tgtDicts, save_srcFile, save_tgtFile, save_factFile, save_persFile, trun=True):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s & %s & %s  ...' % (srcFile, tgtFile, factFile, persFile))
    srcF = open(srcFile, encoding='utf8')
    tgtF = open(tgtFile, encoding='utf8')
    factF = open(factFile, encoding='utf8')
    persF = open(persFile, encoding='utf8')

    srcIdF = open(save_srcFile + '.id', 'w')
    tgtIdF = open(save_tgtFile + '.id', 'w')
    srcStrF = open(save_srcFile + '.str', 'w', encoding='utf8')
    tgtStrF = open(save_tgtFile + '.str', 'w', encoding='utf8')
    factIdF = open(save_factFile + '.id', 'w')
    persIdF = open(save_persFile + '.id', 'w')
    factStrF = open(save_factFile + '.str', 'w', encoding='utf8')
    persStrF = open(save_persFile + '.str', 'w', encoding='utf8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()
        fline = factF.readline()
        pline = persF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        if opt.lower:
            sline = sline.lower()
            tline = tline.lower()
            fline = fline.lower()
            pline = pline.lower()

        sline = sline.strip()
        tline = tline.strip()
        fline = eval(fline)
        pline = eval(pline)

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        srcWords = sline.split() if not opt.src_char else list(sline)
        tgtWords = tline.split() if not opt.tgt_char else list(tline)
        factWords = [f.split() if not opt.src_char else list(f) for f in fline]
        persWords = [p.split() if not opt.src_char else list(p) for p in pline]

        if (opt.src_filter == 0 or len(sline.split()) <= opt.src_filter) and \
           (opt.tgt_filter == 0 or len(tline.split()) <= opt.tgt_filter):

            if opt.src_trun > 0 and trun:
                srcWords = srcWords[:opt.src_trun]
            if opt.tgt_trun > 0 and trun:
                tgtWords = tgtWords[:opt.tgt_trun]

            srcIds = srcDicts.convertToIdx(srcWords, utils.UNK_WORD)
            tgtIds = tgtDicts.convertToIdx(
                tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)
            factIds = [srcDicts.convertToIdx(f, utils.UNK_WORD) for f in factWords]
            persIds = [srcDicts.convertToIdx(p, utils.UNK_WORD) for p in persWords]

            #persBow = srcDicts.convertToBow([y for x in persIds for y in x])

            srcIdF.write(" ".join(list(map(str, srcIds)))+'\n')
            tgtIdF.write(" ".join(list(map(str, tgtIds)))+'\n')
            factIdF.write(str(factIds)+'\n')
            persIdF.write(str(persIds)+'\n')
            if not opt.src_char:
                srcStrF.write(" ".join(srcWords)+'\n')
                factStrF.write(str([" ".join(f) for f in factWords])+'\n')
                persStrF.write(str([" ".join(p) for p in persWords])+'\n')
            else:
                srcStrF.write("".join(srcWords) + '\n')
                factStrF.write(str(["".join(f) for f in factWords])+'\n')
                persStrF.write(str(["".join(p) for p in persWords])+'\n')
            if not opt.tgt_char:
                tgtStrF.write(" ".join(tgtWords)+'\n')
            else:
                tgtStrF.write("".join(tgtWords) + '\n')

            sizes += 1
        else:
            limit_ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    srcStrF.close()
    tgtStrF.close()
    srcIdF.close()
    tgtIdF.close()
    factF.close()
    persF.close()
    factStrF.close()
    persStrF.close()
    factIdF.close()
    persIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {'srcF': save_srcFile + '.id', 'tgtF': save_tgtFile + '.id', 'factF': save_factFile + '.id', 'persF': save_persFile + '.id',
            'original_srcF': save_srcFile + '.str', 'original_tgtF': save_tgtFile + '.str',
            'length': sizes}


def main():

    dicts = {}

    train_src, train_tgt = opt.load_data + 'train.' + \
        opt.src_suf, opt.load_data + 'train.' + opt.tgt_suf
    valid_src, valid_tgt = opt.load_data + 'valid.' + \
        opt.src_suf, opt.load_data + 'valid.' + opt.tgt_suf
    test_src, test_tgt = opt.load_data + 'test.' + \
        opt.src_suf, opt.load_data + 'test.' + opt.tgt_suf
    train_fact, train_pers = opt.load_data + 'train.' + \
        opt.fact_suf, opt.load_data + 'train.' + opt.pers_suf
    valid_fact, valid_pers = opt.load_data + 'valid.' + \
        opt.fact_suf, opt.load_data + 'valid.' + opt.pers_suf
    test_fact, test_pers = opt.load_data + 'test.' + \
        opt.fact_suf, opt.load_data + 'test.' + opt.pers_suf

    save_train_src, save_train_tgt = opt.save_data + \
        'train.src', opt.save_data + 'train.tgt'
    save_valid_src, save_valid_tgt = opt.save_data + \
        'valid.src', opt.save_data + 'valid.tgt'
    save_test_src, save_test_tgt = opt.save_data + \
        'test.src', opt.save_data + 'test.tgt'
    save_train_fact, save_train_pers = opt.save_data + \
        'train.fact', opt.save_data + 'train.pers'
    save_valid_fact, save_valid_pers = opt.save_data + \
        'valid.fact', opt.save_data + 'valid.pers'
    save_test_fact, save_test_pers = opt.save_data + \
        'test.fact', opt.save_data + 'test.pers'

    src_dict, tgt_dict, bow_dict = opt.save_data + 'src.dict', opt.save_data + 'tgt.dict', opt.save_data + 'bow.dict'

    if opt.share:
        assert opt.src_vocab_size == opt.tgt_vocab_size
        print('Building source and target vocabulary...')
        dicts['src'] = dicts['tgt'] = utils.Dict(
            [utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=opt.lower)
        dicts['src'] = makeVocabulary(
            train_src, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
        dicts['src'] = makeVocabulary(
            train_fact, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
        dicts['src'] = makeVocabulary(
            train_pers, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
        dicts['src'] = dicts['tgt'] = makeVocabulary(
            train_tgt, opt.tgt_trun, opt.tgt_filter, opt.tgt_char, dicts['src'], opt.tgt_vocab_size, opt.bow_vocab_size, stopwords=opt.stopwords, freq=opt.freq)
    else:
        print('Building source vocabulary...')
        if opt.src_dict:
            dicts['src'] = utils.Dict(data=opt.src_dict, lower=opt.lower)
        else:
            dicts['src'] = utils.Dict(
                [utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=opt.lower)
            dicts['src'] = makeVocabulary(
                train_src, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
            dicts['src'] = makeVocabulary(
                train_fact, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
            dicts['src'] = makeVocabulary(
                train_pers, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size, opt.bow_vocab_size, stopwords=opt.stopwords, freq=opt.freq)
        print('Building target vocabulary...')
        if opt.tgt_dict:
            dicts['tgt'] = utils.Dict(data=opt.tgt_dict, lower=opt.lower)
        else:
            dicts['tgt'] = utils.Dict(
                [utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=opt.lower)
            dicts['tgt'] = makeVocabulary(
                train_tgt, opt.tgt_trun, opt.tgt_filter, opt.tgt_char, dicts['tgt'], opt.tgt_vocab_size, freq=opt.freq)

    print('Preparing training ...')
    train = makeData(train_src, train_tgt, train_fact, train_pers,
                     dicts['src'], dicts['tgt'], save_train_src, save_train_tgt, save_train_fact, save_train_pers)

    print('Preparing validation ...')
    valid = makeData(valid_src, valid_tgt, valid_fact, valid_pers,
                     dicts['src'], dicts['tgt'], save_valid_src, save_valid_tgt, save_valid_fact, save_valid_pers, trun=False)

    print('Preparing test ...')
    test = makeData(test_src, test_tgt, test_fact, test_pers,
                    dicts['src'], dicts['tgt'], save_test_src, save_test_tgt, save_test_fact, save_test_pers, trun=False)

    print('Saving source vocabulary to \'' + src_dict + '\'...')
    dicts['src'].writeFile(src_dict)

    print('Saving bow vocabulary to \'' + bow_dict + '\'...')
    dicts['src'].writeBowFile(bow_dict)

    print('Saving source vocabulary to \'' + tgt_dict + '\'...')
    dicts['tgt'].writeFile(tgt_dict)

    data = {'train': train, 'valid': valid,
            'test': test, 'dict': dicts}
    pickle.dump(data, open(opt.save_data+'data.pkl', 'wb'))


if __name__ == "__main__":
    main()

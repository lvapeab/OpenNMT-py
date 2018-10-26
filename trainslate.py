""" Translate and trains a model """
from __future__ import division, unicode_literals
import argparse
import codecs

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
import os
import signal
import torch

import onmt.opts as opts

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
from onmt.utils.logging import logger

from onmt.translate.translator import Translator
from onmt.translate import TranslationBuilder
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, build_dataset, \
    _load_fields, _collect_report_features, OrderedIterator
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.train_single import training_opt_postprocessing, _tally_parameters, _check_save_model_path


def main(opt, device_id):
    # opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        raise Exception('You need to load a model')

    logger.info('Loading data from %s' % opt.data)
    dataset = next(lazily_load_dataset("train", opt))
    data_type = dataset.data_type
    logger.info('Data type %s' % data_type)

    # Load fields generated from preprocess phase.
    fields = _load_fields(dataset, data_type, opt, checkpoint)
    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    dataset_iter = build_dataset_iter(lazily_load_dataset("train", opt), fields, opt)
    out_file = codecs.open(opt.output, 'w+', 'utf-8')
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    translation_builder = TranslationBuilder(dataset, fields,
                                             n_best=opt.n_best,
                                             replace_unk=opt.replace_unk,
                                             has_tgt=False)

    def train_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("train", opt), fields, opt)

    trainer = build_trainer(opt, device_id, model, fields,
                            optim, data_type, model_saver=model_saver)

    translator = Translator(trainer.model, fields, opt.beam_size, global_scorer=scorer,
                            out_file=out_file, report_score=False,
                            copy_attn=model_opt.copy_attn, logger=logger)

    for i, batch in enumerate(dataset_iter):
        unprocessed_translations = translator.translate_batch(batch, dataset)
        translations = translation_builder.from_batch(unprocessed_translations)
        print "Translations: ", ' '.join(translations[0].pred_sents[0])
        trainer.train_from_data(batch, train_steps=1)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='trainslate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.trainslate_opts(parser)

    opt = parser.parse_args()
    main(opt, 0)

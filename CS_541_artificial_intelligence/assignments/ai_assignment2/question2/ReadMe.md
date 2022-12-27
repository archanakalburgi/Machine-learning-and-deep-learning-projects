# Obeservations
```
(base) jovyan@8ef4416ed7d3:~/work/data/archana/artificial_intelegence/Assignment2/neural_machine_translation$ sacrebleu data/test.eng < test_beam_1.out 
sacreBLEU: That's 100 lines that end in a tokenized period ('.')
sacreBLEU: It looks like you forgot to detokenize your test data, which may hurt your score.
sacreBLEU: If you insist your data is detokenized, or don't care, you can suppress this message with the `force` parameter.
{
 "name": "BLEU",
 "score": 12.4,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "41.2/13.3/7.8/5.5 (BP = 1.000 ratio = 1.035 hyp_len = 6438 ref_len = 6218)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}
(base) jovyan@8ef4416ed7d3:~/work/data/archana/artificial_intelegence/Assignment2/neural_machine_translation$ 
```
import sacrebleu

# 定义参考句子和翻译结果
reference = "This is a reference sentence."
translation = "This is a translation sentence."

# 使用 sacrebleu.corpus_chrf 函数计算 CHRF 指标
chrf_score = sacrebleu.corpus_chrf(translation, [reference])

print("CHRF Score:", chrf_score.score)


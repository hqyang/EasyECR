# corefprompt
1. 执行trigger_detection.py脚本在三个数据集（kbpmix, ace2005, mavenere）完成模型的训练，预测结果在三个数据集的测试集（验证集）上得到结果。
2. 执行sample_simi.py脚本在三个数据集（kbpmix, ace2005, mavenere）完成模型的训练，预测结果在三个数据集的测试集（验证集）上得到结果。
3. 结合上面两个步骤得到的结果，执行related_info_extraction.py脚本，整合为新的ecr_data，进一步用于模型的训练。
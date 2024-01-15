cd .\run_fill_in_blank
python eval.py --input "F:\Me\ML\IconQA\mount\data" --inspect-att True --model patch_transformer_ques_bert --label exp_paper
python eval.py --input "F:\Me\ML\IconQA\mount\data" --inspect-att True --debug True --model patch_transformer_ques_bert --label exp_paper --test_ids '20' '48' '62' '103' '114' '47103'
python eval.py --input "F:\Me\ML\IconQA\mount\data" --inspect-att True --debug True --model patch_transformer_ques_bert --label exp_paper --test_ids '103'

jupyter lab --notebook-dir=D:/Me
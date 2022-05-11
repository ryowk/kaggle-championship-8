MODEL_DIRS=`ls -d */ | grep _model`

for model_dir in $MODEL_DIRS
do
  echo $model_dir
  aws s3 sync s3://pigimaru-kaggle-days-2021/championship-1/${model_dir} ${model_dir}output/
done

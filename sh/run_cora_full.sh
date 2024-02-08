cd ../

gpu=$1
dataset="cora_full"
lt_setting="1"
cls_og="MLP"
ep="10000"
ep_early="1000"
ep_warmup="0"
dropout="0.0"
layer="gcn"
nhid="128"
act='relu'
wght="0.01"
tmpt="1.0"
ndpth="3"
pool_ratios="1.0 0.5 0.5"
embedder="HierTail"
outdir=results/baseline/natural/${dataset}

if [ ! -d ${outdir} ];then
  mkdir ${outdir}
fi

echo "Starts..."
log1=${outdir}/${embedder}-${ndpth}-layers_wgt-${wght}_tmpt-${tmpt}_act-${act}_dropout-${dropout}_nhid-${nhid}.log
python main.py --dataset ${dataset} --lt_setting ${lt_setting} --cls_og ${cls_og} --ep ${ep} --ep_early ${ep_early} \
   --dropout ${dropout} --layer ${layer} --gpu ${gpu} --nhid ${nhid} --ndpth ${ndpth} --pool_ratios ${pool_ratios} \
   --activation ${act} --weight_cpc ${wght} --temperature ${tmpt} --embedder ${embedder} >>${log1} 2>&1
echo "Ends."
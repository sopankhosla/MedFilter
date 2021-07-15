import sys; sys.path.append('../common'); 
from helper import *
from bert_serving.client import BertClient

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='MedFilter')

	parser.add_argument("--embed", 	 	 default="bert_ft_new", 	type=str, 		help=" ")
	parser.add_argument("--pool", 	 	 default="mean_sec_last", 	type=str, 		help=" ")
	parser.add_argument("--max_seq_len", 	 default=64, 			type=int, 		help=" ")
	parser.add_argument("--num_proc", 	 default=16, 			type=int, 		help=" ")
	parser.add_argument("--data_dir",  	 default="/data/",		help='Directory containing dataset')

	args = parser.parse_args()

	embed_map = {}
	all_data  = []
	for line in open(args.data_dir + 'main.json'):
		conv			= json.loads(line)
		_id 			= conv['meta']['id']
		_, conv['transcript']	= zip(*sorted(conv['transcript'].items(), key = lambda x: int(x[0])))
		all_data.append((_id, conv['transcript']))

	def process_data(pid, trans_list):
		bc = BertClient(check_length=False)
		result = []
		for i, (_id, trans) in enumerate(trans_list):
			embed = []
			for chunk in getChunks(trans, 64):
				embed.append(bc.encode([x['txt'] for x in chunk]))

			result.append((_id, np.concatenate(embed, axis=0)))
			if i % 100 == 0: print('Completed [{}] {}, {}'.format(pid, i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))

		print('All jobs Over!')

		return result

	num_procs = args.num_proc
	chunks    = partition(all_data, num_procs)
	data_list = Parallel(n_jobs = num_procs)(delayed(process_data)(i, chunk) for i, chunk in enumerate(chunks))
	embed_map = dict(mergeList(data_list))

	dump_dir = '{}/embeddings/{}'.format(args.data_dir, args.embed)
	os.system('mkdir -p {}'.format(dump_dir))
	for _id, embed in embed_map.items():
		pickle.dump({'embeddings': embed}, open('{}/{}.pkl'.format(dump_dir, _id), 'wb'))


"""
python ./transformers/src/transformers/convert_bert_pytorch_checkpoint_to_original_tf.py \
	--model_name bert_model \
	--pytorch_model_path ./bert_models/bert_plain_model_28_03_2020_20:39:24/pytorch.bin \
	--tf_cache_dir ./bert_models/bert_plain_model_28_03_2020_20:39:24/tf_model
cd ./bert_models/bert_plain_model_28_03_2020_20:39:24/tf_model/
cp <dir>/vocab.txt .
cp <dir>/bert_config.json .
mv bert_base_uncased.ckpt.index bert_model.ckpt.index
mv bert_base_uncased.ckpt.meta  bert_model.ckpt.meta
mv bert_base_uncased.ckpt.data-00000-of-00001 bert_model.ckpt.data-00000-of-00001
cd .. 
bert-serving-start -model_dir ./tf_model/ -num_worker=2 -device_map 3 -max_seq_len 64
"""
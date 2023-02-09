import os
import sys

all_log_fol = sys.argv[1]
BASE_DIR = all_log_fol
all_log_fol = sorted(os.listdir(all_log_fol))

to_keep = ["log_0928_all_dataparallel", "log_0926_bigsubset_dataparallel_corrected"]

num_deleted = 0
for log_file in all_log_fol:
	if not os.path.isdir(os.path.join(BASE_DIR, log_file)):
		continue

	if "log" not in log_file:
		continue

	all_files = sorted(os.listdir(os.path.join(BASE_DIR, log_file)))
	all_checkpoints = [x for x in all_files if 'checkpoint' in x]
	
	if log_file in to_keep:
		print("Keeping "+log_file)
		continue
		# ##Remove all except the last 2 checkpoints
		# for i in range(len(all_checkpoints)-2):
		# 	file_to_delete = os.path.join(BASE_DIR, log_file, all_checkpoints[i])
		# 	os.system('rm ' + file_to_delete)
		# 	print("Deleted: "+ file_to_delete)
		# 	num_deleted += 1

	else:
		os.system('rm -rf ' + log_file)
		print("Deleted: "+ log_file)
		num_deleted += 1

print()
print("Deleted "+str(num_deleted)+" files.")
print("Done.")
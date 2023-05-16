import os
import sys

all_log_fol = sys.argv[1]
BASE_DIR = all_log_fol
all_log_fol = sorted(os.listdir(all_log_fol))

num_deleted = 0
for log_file in all_log_fol:
	if not os.path.isdir(os.path.join(BASE_DIR, log_file)):
		continue

	if "log" not in log_file:
		continue

	all_fols = sorted(os.listdir(os.path.join(BASE_DIR, log_file)))
	print(log_file)

	for fol in all_fols:
		if not os.path.isdir(os.path.join(BASE_DIR, log_file, fol)):
			continue		
		all_files = sorted(os.listdir(os.path.join(BASE_DIR, log_file, fol)))
		all_checkpoints = [x for x in all_files if '.tar' in x]

		##Remove all except the last 2 checkpoints
		for i in range(len(all_checkpoints)-2):
			file_to_delete = os.path.join(BASE_DIR, log_file, fol, all_checkpoints[i])
			os.system('rm ' + file_to_delete)
			print("Deleted: "+ file_to_delete)
			num_deleted += 1


print()
print("Deleted "+str(num_deleted)+" files.")
print("Done.")
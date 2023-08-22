echo "START VALID --->"
python run_valid.py --ckpt loss --cam --batch_size 4
python run_valid.py --ckpt score --cam --batch_size 4
echo "<--- END VALID"

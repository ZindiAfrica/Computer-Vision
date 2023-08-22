echo "START TEST PREPROCESS --->"
python run_test_preprocess.py --config_name config.yaml
echo "<--- END TEST PREPROCESS"

echo "START TEST"
python run_test.py --ckpt loss
python run_test.py --ckpt score
echo "<--- END TEST"

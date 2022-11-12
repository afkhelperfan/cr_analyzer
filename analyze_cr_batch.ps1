$boss = $args[0]
$user = $args[1]
$lang = $args[2]
[Int] $start = $args[3]
[Int] $end = $args[4]
$thres = $args[5]
$viz = $args[6]

for ($i=$start; $i -lt $end + 1; $i++){
    python3 cr_analyzer.py $boss $user $i $lang $thres $viz
}
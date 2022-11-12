
$boss = $args[0]
$user = $args[1]
[Int] $start = $args[2]
[Int] $end = $args[3]

for ($i=$start; $i -lt $end + 1; $i++){
    $count = 1
    write-host "data/$boss/$user/$i"
    Get-ChildItem -Path "data/$boss/$user/$i" -Filter '*.png' -File |
    Sort-Object LastWriteTime |
    Rename-Item -NewName { '{0:D1}{1}' -f $script:count++, $_.Extension }
}
  




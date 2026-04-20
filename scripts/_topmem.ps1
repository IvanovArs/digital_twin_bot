Get-Process | Sort-Object -Property WorkingSet -Descending | Select-Object -First 15 -Property Id,ProcessName,@{Name='MB';Expression={[int]($_.WorkingSet/1MB)}} | Format-Table -AutoSize

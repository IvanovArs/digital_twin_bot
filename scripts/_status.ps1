Get-Process llama-server -ErrorAction SilentlyContinue |
  Select-Object Id, @{Name='MB'; Expression={[int]($_.WorkingSet/1MB)}} |
  Format-Table -AutoSize
Write-Host '---listen---'
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalPort -eq 8089 -or $_.LocalPort -eq 8080 } |
  Select-Object LocalAddress, LocalPort, OwningProcess |
  Format-Table -AutoSize

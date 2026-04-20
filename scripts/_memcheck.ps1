$os = Get-CimInstance Win32_OperatingSystem
Write-Host ("FreePhysMB=" + [int]($os.FreePhysicalMemory/1024) + " TotalPhysMB=" + [int]($os.TotalVisibleMemorySize/1024))
Write-Host ("FreeVirtMB=" + [int]($os.FreeVirtualMemory/1024) + " TotalVirtMB=" + [int]($os.TotalVirtualMemorySize/1024))

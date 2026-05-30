param(
    [string]$OutputDir = "data\raw",
    [string]$PhrasesFile = "data\phrases.txt"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $root $OutputDir
$phrasePath = Join-Path $root $PhrasesFile
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

Add-Type -AssemblyName System.Speech
$phrases = Get-Content $phrasePath | Where-Object { $_.Trim().Length -gt 0 }
$manifestRows = @("audio_path,transcript")

foreach ($phrase in $phrases) {
    for ($i = 1; $i -le 5; $i++) {
        $safe = $phrase.ToLower().Replace(" ", "_")
        $fileName = "{0}_sapi_{1:D2}.wav" -f $safe, $i
        $wavPath = Join-Path $outputPath $fileName
        $speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $speaker.Rate = ($i % 3) - 1
        $speaker.Volume = 100
        $speaker.SetOutputToWaveFile($wavPath)
        $speaker.Speak($phrase)
        $speaker.Dispose()
        $relative = "data/raw/$fileName"
        $manifestRows += "$relative,$phrase"
    }
}

$manifestRows | Set-Content -Encoding UTF8 (Join-Path $outputPath "manifest.csv")
Write-Host "Created SAPI speech dataset in $outputPath"

param([ValidateSet('up','ps','logs')][string]$Action='ps')
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeArgs = @('compose','-f','docker-compose.yml','-f','docker-compose.release.yml',
    '-f','docker-compose.injury-chain.yml','-f','docker-compose.paired-models.yml',
    '-f','docker-compose.win-shadow.yml','-f','docker-compose.fixed-mlp.yml')
Push-Location -LiteralPath $projectRoot
try {
    if ($Action -eq 'up') { & docker @composeArgs up -d --no-build }
    elseif ($Action -eq 'logs') { & docker @composeArgs logs --tail 100 }
    else { & docker @composeArgs ps }
    if ($LASTEXITCODE -ne 0) { throw 'Production compose command failed' }
} finally { Pop-Location }

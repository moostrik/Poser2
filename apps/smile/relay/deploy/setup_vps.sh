#!/usr/bin/env bash
# The relay's machine from a fresh Ubuntu VPS, once; safe to run again (everything in place is kept).
# From the PC, as the login user TransIP made (it has sudo):
#     C:\Windows\System32\OpenSSH\ssh.exe moostrik@136.144.146.162 bash -s < apps\smile\relay\deploy\setup_vps.sh
# It makes: the hostname, the clock (NTP), the updates (now and unattended), SSH by keys only, the firewall
# (22, 80, 443), Caddy (TLS for smile.oostrik.net, the proxy to the relay), aiohttp from Ubuntu's packages,
# the user `smile` (no login), the relay as a systemd service bound to 127.0.0.1:8800. The relay's code, token
# and library come after it with apps\smile\scripts\deploy_relay.bat; the service starts once they are there.
set -euo pipefail

NAME=smile.oostrik.net
RELAY_USER=smile
HOME_DIR=/home/$RELAY_USER
export DEBIAN_FRONTEND=noninteractive

echo "== the machine"
sudo hostnamectl set-hostname smile
sudo timedatectl set-ntp true
sudo apt-get update -q
sudo apt-get upgrade -yq
sudo apt-get install -yq python3 python3-aiohttp ufw unattended-upgrades curl gnupg
sudo dpkg-reconfigure -f noninteractive unattended-upgrades

echo "== ssh: keys only"
sudo sed -i -E 's/^#?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl reload ssh

echo "== firewall: 22, 80, 443"
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

echo "== caddy"
if ! command -v caddy >/dev/null; then
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    sudo apt-get update -q
    sudo apt-get install -yq caddy
fi
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
# SMILE Online: TLS for the phones (Let's Encrypt, renewed by Caddy), everything to the relay on this machine
$NAME {
    reverse_proxy 127.0.0.1:8800
}
EOF
sudo systemctl enable caddy >/dev/null
sudo systemctl restart caddy

echo "== the user $RELAY_USER"
id -u $RELAY_USER >/dev/null 2>&1 || sudo adduser --disabled-password --gecos 'SMILE relay' $RELAY_USER
sudo -u $RELAY_USER mkdir -p $HOME_DIR/relay/data
python3 -c 'import aiohttp; print("aiohttp", aiohttp.__version__)'

echo "== the service"
sudo tee /etc/systemd/system/smile-relay.service >/dev/null <<EOF
[Unit]
Description=SMILE Online relay (apps/smile/relay)
After=network-online.target
Wants=network-online.target

[Service]
User=$RELAY_USER
WorkingDirectory=$HOME_DIR
ExecStart=/usr/bin/python3 -m relay.server --host 127.0.0.1 --port 8800
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable smile-relay >/dev/null
if [ -f $HOME_DIR/relay/server.py ] && [ -f $HOME_DIR/relay/token.txt ]; then
    sudo systemctl restart smile-relay
    echo "relay: $(systemctl is-active smile-relay)"
else
    echo "relay: not deployed yet (apps\\smile\\scripts\\deploy_relay.bat)"
fi
echo "== done: $(hostname), $(date -u '+%Y-%m-%d %H:%M UTC'), caddy $(systemctl is-active caddy)"

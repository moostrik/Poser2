"""Network validation utilities for OSC and ArtNet connections."""

import socket
import ipaddress
import subprocess


def validate_network(ip_address: str, port: int) -> bool:
    """Validate network connectivity by attempting socket connection.

    Args:
        ip_address: IP address to test
        port: Port number to test

    Returns:
        True if network route is available, False otherwise
    """
    try:
        # Create a UDP socket and attempt to connect
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_socket.settimeout(1.0)
        # For UDP, connect() doesn't actually send packets but validates routing
        test_socket.connect((ip_address, port))
        test_socket.close()
        return True
    except socket.error:
        return False
    except Exception:
        return False


def validate_ip(ip_address: str) -> bool:
    """Validate IP address format.

    Args:
        ip_address: IP address string to validate

    Returns:
        True if IP format is valid, False otherwise
    """
    try:
        ipaddress.ip_address(ip_address)
        return True
    except ValueError:
        return False


def ping_ip(ip_address: str) -> bool:
    """Ping IP address to check if host is reachable.

    Args:
        ip_address: IP address to ping

    Returns:
        True if ping successful, False otherwise
    """
    try:
        result = subprocess.run(
            ['ping', '-n', '1', '-w', '500', ip_address],
            capture_output=True,
            timeout=2.0
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def validate_connection(ip_address: str, port: int, class_name: str) -> bool:
    """Validate network connectivity, IP format, and host reachability.

    Performs validation in order:
    1. Network connectivity (ERROR if fails)
    2. IP format validation (ERROR if fails)
    3. Ping test (WARNING if fails, but continues)

    Args:
        ip_address: Target IP address
        port: Target port number
        class_name: Name of calling class for error messages

    Returns:
        False if network or IP validation fails, True otherwise
    """
    # Check network first - if no network, nothing else matters
    if not validate_network(ip_address, port):
        print(f"{class_name} ERROR: Network unreachable to {ip_address}:{port}. Stopping sender thread.")
        return False

    # Check IP format
    if not validate_ip(ip_address):
        print(f"{class_name} ERROR: Invalid IP address format: {ip_address}")
        return False

    # Ping is optional - warn but don't fail
    if not ping_ip(ip_address):
        print(f"{class_name} WARNING: IP {ip_address} is not reachable (ping failed). Device may appear later.")

    return True

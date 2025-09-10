"""SSL certificate management."""

import datetime as _datetime
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


class CertificateManager:
    """Manages SSL certificate generation."""

    def generate_self_signed(
        self, output_dir: Path, domain: str = "localhost"
    ) -> tuple[Path, Path]:
        """Generate self-signed certificate for development."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate private key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Generate certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, domain),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(_datetime.UTC))
            .not_valid_after(datetime.now(_datetime.UTC) + timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(domain),
                        x509.DNSName("localhost"),
                        x509.DNSName("127.0.0.1"),
                    ]
                ),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        # Write files
        cert_path = output_dir / "cert.pem"
        key_path = output_dir / "key.pem"

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_path, "wb") as f:
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        return cert_path, key_path

    def generate_letsencrypt(
        self, domain: str, email: str, output_dir: Optional[Path] = None, staging: bool = False
    ) -> tuple[Path, Path]:
        """Generate Let's Encrypt certificate.

        Args:
        ----
            domain: Domain name for certificate
            email: Email for Let's Encrypt account
            output_dir: Custom output directory (uses /etc/letsencrypt by default)
            staging: Use Let's Encrypt staging server for testing

        """
        cmd = [
            "certbot",
            "certonly",
            "--standalone",
            "--non-interactive",
            "--agree-tos",
            "--email",
            email,
            "-d",
            domain,
        ]

        if staging:
            cmd.append("--staging")

        if output_dir:
            # Use custom config and work directories
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(
                [
                    "--config-dir",
                    str(output_dir),
                    "--work-dir",
                    str(output_dir / "work"),
                    "--logs-dir",
                    str(output_dir / "logs"),
                ]
            )
            cert_base = output_dir / "live" / domain
        else:
            cert_base = Path(f"/etc/letsencrypt/live/{domain}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Certbot failed: {result.stderr}")

        cert_path = cert_base / "fullchain.pem"
        key_path = cert_base / "privkey.pem"

        if not cert_path.exists() or not key_path.exists():
            raise RuntimeError(f"Certificate files not found at {cert_base}")

        return cert_path, key_path

    def get_cert_info(self, cert_path: Path) -> dict:
        """Get information about an existing certificate."""
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())

        return {
            "subject": cert.subject.rfc4514_string(),
            "issuer": cert.issuer.rfc4514_string(),
            "not_before": cert.not_valid_before_utc,
            "not_after": cert.not_valid_after_utc,
            "serial_number": cert.serial_number,
            "is_self_signed": cert.issuer == cert.subject,
        }

    def inspect_certificate(self, cert_path: Path) -> dict:
        """Inspect a certificate (alias for get_cert_info for CLI compatibility)."""
        return self.get_cert_info(cert_path)

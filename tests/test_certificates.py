"""Tests for the certificates utility module."""

import datetime as _datetime
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from caption_flow.utils.certificates import CertificateManager
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


@pytest.fixture
def cert_manager():
    """Certificate manager instance."""
    return CertificateManager()


@pytest.fixture
def temp_output_dir():
    """Temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_cert_data():
    """Sample certificate data for testing."""
    # Generate a test certificate
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "test.example.com"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(_datetime.UTC))
        .not_valid_after(datetime.now(_datetime.UTC) + timedelta(days=30))
        .sign(key, hashes.SHA256())
    )

    return {
        "certificate": cert,
        "key": key,
        "cert_pem": cert.public_bytes(serialization.Encoding.PEM),
        "key_pem": key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    }


class TestCertificateManagerInit:
    """Test CertificateManager initialization."""

    def test_init(self):
        """Test CertificateManager can be instantiated."""
        manager = CertificateManager()
        assert manager is not None


class TestGenerateSelfSigned:
    """Test self-signed certificate generation."""

    def test_generate_self_signed_default_domain(self, cert_manager, temp_output_dir):
        """Test generating self-signed certificate with default domain."""
        cert_path, key_path = cert_manager.generate_self_signed(temp_output_dir)

        # Check files were created
        assert cert_path.exists()
        assert key_path.exists()
        assert cert_path == temp_output_dir / "cert.pem"
        assert key_path == temp_output_dir / "key.pem"

        # Check files have content
        assert cert_path.stat().st_size > 0
        assert key_path.stat().st_size > 0

    def test_generate_self_signed_custom_domain(self, cert_manager, temp_output_dir):
        """Test generating self-signed certificate with custom domain."""
        domain = "example.com"
        cert_path, key_path = cert_manager.generate_self_signed(temp_output_dir, domain)

        # Check files were created
        assert cert_path.exists()
        assert key_path.exists()

        # Verify certificate contains correct domain
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())

        # Check subject common name
        subject_cn = None
        for attribute in cert.subject:
            if attribute.oid == NameOID.COMMON_NAME:
                subject_cn = attribute.value
                break

        assert subject_cn == domain

    def test_generate_self_signed_creates_directory(self, cert_manager):
        """Test that output directory is created if it doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        try:
            output_dir = Path(temp_dir) / "new_dir" / "nested"
            assert not output_dir.exists()

            cert_path, key_path = cert_manager.generate_self_signed(output_dir)

            assert output_dir.exists()
            assert cert_path.exists()
            assert key_path.exists()
        finally:
            shutil.rmtree(temp_dir)

    def test_generate_self_signed_certificate_validity(self, cert_manager, temp_output_dir):
        """Test that generated certificate has correct validity period."""
        cert_path, key_path = cert_manager.generate_self_signed(temp_output_dir)

        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())

        # Check validity period (should be ~365 days)
        validity_period = cert.not_valid_after_utc - cert.not_valid_before_utc
        assert 364 <= validity_period.days <= 366  # Allow some tolerance

    def test_generate_self_signed_subject_alternative_names(self, cert_manager, temp_output_dir):
        """Test that certificate includes subject alternative names."""
        cert_path, key_path = cert_manager.generate_self_signed(temp_output_dir, "example.com")

        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())

        # Find SAN extension
        san_ext = None
        for ext in cert.extensions:
            if isinstance(ext.value, x509.SubjectAlternativeName):
                san_ext = ext.value
                break

        assert san_ext is not None
        dns_names = [name.value for name in san_ext if isinstance(name, x509.DNSName)]
        assert "example.com" in dns_names
        assert "localhost" in dns_names
        assert "127.0.0.1" in dns_names


class TestGenerateLetsEncrypt:
    """Test Let's Encrypt certificate generation."""

    @patch("subprocess.run")
    def test_generate_letsencrypt_success(self, mock_subprocess, cert_manager, temp_output_dir):
        """Test successful Let's Encrypt certificate generation."""
        # Mock successful certbot execution
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # Create expected certificate files
        cert_dir = temp_output_dir / "live" / "example.com"
        cert_dir.mkdir(parents=True)
        (cert_dir / "fullchain.pem").touch()
        (cert_dir / "privkey.pem").touch()

        cert_path, key_path = cert_manager.generate_letsencrypt(
            domain="example.com", email="test@example.com", output_dir=temp_output_dir
        )

        assert cert_path == cert_dir / "fullchain.pem"
        assert key_path == cert_dir / "privkey.pem"

        # Check certbot was called correctly
        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        assert "certbot" in cmd
        assert "example.com" in cmd
        assert "test@example.com" in cmd

    @patch("subprocess.run")
    def test_generate_letsencrypt_staging(self, mock_subprocess, cert_manager, temp_output_dir):
        """Test Let's Encrypt certificate generation with staging."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # Create expected certificate files
        cert_dir = temp_output_dir / "live" / "example.com"
        cert_dir.mkdir(parents=True)
        (cert_dir / "fullchain.pem").touch()
        (cert_dir / "privkey.pem").touch()

        cert_manager.generate_letsencrypt(
            domain="example.com", email="test@example.com", output_dir=temp_output_dir, staging=True
        )

        # Check that --staging was added to command
        cmd = mock_subprocess.call_args[0][0]
        assert "--staging" in cmd

    @patch("subprocess.run")
    def test_generate_letsencrypt_default_output_dir(self, mock_subprocess, cert_manager):
        """Test Let's Encrypt generation with default output directory."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # Mock the default certificate path
        with patch("pathlib.Path.exists", return_value=True):
            cert_path, key_path = cert_manager.generate_letsencrypt(
                domain="example.com", email="test@example.com"
            )

        # Should use default path
        assert str(cert_path) == "/etc/letsencrypt/live/example.com/fullchain.pem"
        assert str(key_path) == "/etc/letsencrypt/live/example.com/privkey.pem"

        # Should not include custom config directories in command
        cmd = mock_subprocess.call_args[0][0]
        assert "--config-dir" not in cmd

    @patch("subprocess.run")
    def test_generate_letsencrypt_certbot_failure(
        self, mock_subprocess, cert_manager, temp_output_dir
    ):
        """Test handling of certbot failure."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Certificate validation failed"

        with pytest.raises(RuntimeError) as exc_info:
            cert_manager.generate_letsencrypt(
                domain="example.com", email="test@example.com", output_dir=temp_output_dir
            )

        assert "Certbot failed" in str(exc_info.value)
        assert "Certificate validation failed" in str(exc_info.value)

    @patch("subprocess.run")
    def test_generate_letsencrypt_missing_files(
        self, mock_subprocess, cert_manager, temp_output_dir
    ):
        """Test handling when certificate files are not created."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # Don't create the expected files
        with pytest.raises(RuntimeError) as exc_info:
            cert_manager.generate_letsencrypt(
                domain="example.com", email="test@example.com", output_dir=temp_output_dir
            )

        assert "Certificate files not found" in str(exc_info.value)


class TestGetCertInfo:
    """Test certificate information extraction."""

    def test_get_cert_info_self_signed(self, cert_manager, temp_output_dir, sample_cert_data):
        """Test getting info from self-signed certificate."""
        # Write sample certificate to file
        cert_path = temp_output_dir / "test.pem"
        with open(cert_path, "wb") as f:
            f.write(sample_cert_data["cert_pem"])

        info = cert_manager.get_cert_info(cert_path)

        assert "subject" in info
        assert "issuer" in info
        assert "not_before" in info
        assert "not_after" in info
        assert "serial_number" in info
        assert "is_self_signed" in info

        # For self-signed cert, subject should equal issuer
        assert info["is_self_signed"]
        assert "test.example.com" in info["subject"]

    def test_get_cert_info_real_certificate(self, cert_manager, temp_output_dir):
        """Test getting info from actual generated certificate."""
        cert_path, _ = cert_manager.generate_self_signed(temp_output_dir, "mytest.com")

        info = cert_manager.get_cert_info(cert_path)

        assert isinstance(info["not_before"], datetime)
        assert isinstance(info["not_after"], datetime)
        assert isinstance(info["serial_number"], int)
        assert info["is_self_signed"]
        assert "mytest.com" in info["subject"]

    def test_get_cert_info_file_not_found(self, cert_manager, temp_output_dir):
        """Test handling when certificate file doesn't exist."""
        non_existent_path = temp_output_dir / "nonexistent.pem"

        with pytest.raises(FileNotFoundError):
            cert_manager.get_cert_info(non_existent_path)

    def test_get_cert_info_invalid_certificate(self, cert_manager, temp_output_dir):
        """Test handling invalid certificate file."""
        invalid_cert_path = temp_output_dir / "invalid.pem"
        with open(invalid_cert_path, "w") as f:
            f.write("This is not a valid certificate")

        with pytest.raises(Exception):  # Should raise cryptography exception
            cert_manager.get_cert_info(invalid_cert_path)


class TestCertificateManagerIntegration:
    """Integration tests for CertificateManager."""

    def test_generate_and_inspect_cycle(self, cert_manager, temp_output_dir):
        """Test generating a certificate and then inspecting it."""
        domain = "integration.test"

        # Generate certificate
        cert_path, key_path = cert_manager.generate_self_signed(temp_output_dir, domain)

        # Inspect certificate
        info = cert_manager.get_cert_info(cert_path)

        # Verify information is consistent
        assert domain in info["subject"]
        assert info["is_self_signed"]
        assert info["not_before"] < info["not_after"]

        # Verify certificate is currently valid (handle timezone awareness)
        now = datetime.now(_datetime.UTC)
        # The certificate dates might be naive, so compare carefully
        not_before = info["not_before"]
        not_after = info["not_after"]

        # Make now timezone-naive for comparison if cert dates are naive
        if not_before.tzinfo is None:
            now = now.replace(tzinfo=None)

        assert not_before <= now <= not_after

    def test_multiple_certificates_same_directory(self, cert_manager, temp_output_dir):
        """Test generating multiple certificates in the same directory."""
        domains = ["test1.com", "test2.com"]
        certificates = []

        for domain in domains:
            domain_dir = temp_output_dir / domain
            cert_path, key_path = cert_manager.generate_self_signed(domain_dir, domain)
            certificates.append((cert_path, key_path, domain))

        # Verify all certificates were created and are different
        for cert_path, key_path, domain in certificates:
            assert cert_path.exists()
            assert key_path.exists()

            info = cert_manager.get_cert_info(cert_path)
            assert domain in info["subject"]

        # Verify certificates have different serial numbers
        serial_numbers = []
        for cert_path, _, _ in certificates:
            info = cert_manager.get_cert_info(cert_path)
            serial_numbers.append(info["serial_number"])

        assert len(set(serial_numbers)) == len(serial_numbers)  # All unique


if __name__ == "__main__":
    pytest.main([__file__])

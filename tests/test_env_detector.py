"""Tests for hardware environment detection module."""

from unittest.mock import Mock, patch

from src.env_detector import (
    detect_environment,
    detect_gpu_info,
    detect_mps_available,
    detect_system_info,
)


class TestDetectEnvironment:
    """Tests for detect_environment function."""

    @patch("src.env_detector.platform.system", return_value="Darwin")
    def test_detect_environment_mac(self, mock_system: Mock) -> None:
        result = detect_environment()
        assert result == "mac"

    @patch("src.env_detector.TORCH_AVAILABLE", True)
    @patch("src.env_detector.platform.system", return_value="Windows")
    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_environment_windows_with_cuda(
        self, mock_cuda_available: Mock, mock_system: Mock
    ) -> None:
        result = detect_environment()
        assert result == "windows_gpu"

    @patch("src.env_detector.TORCH_AVAILABLE", True)
    @patch("src.env_detector.platform.system", return_value="Windows")
    @patch("torch.cuda.is_available", return_value=False)
    def test_detect_environment_windows_no_cuda(
        self, mock_cuda_available: Mock, mock_system: Mock
    ) -> None:
        result = detect_environment()
        assert result == "mac"

    @patch("src.env_detector.TORCH_AVAILABLE", False)
    @patch("src.env_detector.platform.system", return_value="Windows")
    def test_detect_environment_no_torch(self, mock_system: Mock) -> None:
        result = detect_environment()
        assert result == "mac"


class TestDetectGpuInfo:
    """Tests for detect_gpu_info function."""

    @patch("src.env_detector.TORCH_AVAILABLE", True)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 4070 Ti")
    @patch("torch.cuda.get_device_properties")
    def test_detect_gpu_info_with_cuda(
        self,
        mock_get_props: Mock,
        mock_get_name: Mock,
        mock_cuda_available: Mock,
    ) -> None:
        mock_get_props.return_value = Mock(total_mem=12 * 1024 * 1024 * 1024)  # 12 GB

        gpu_name, gpu_vram_mb = detect_gpu_info()

        assert gpu_name == "NVIDIA GeForce RTX 4070 Ti"
        assert gpu_vram_mb == 12 * 1024  # 12288 MB

    @patch("src.env_detector.TORCH_AVAILABLE", False)
    def test_detect_gpu_info_no_cuda(self) -> None:
        gpu_name, gpu_vram_mb = detect_gpu_info()
        assert gpu_name is None
        assert gpu_vram_mb is None


class TestDetectMpsAvailable:
    """Tests for detect_mps_available function."""

    @patch("src.env_detector.TORCH_AVAILABLE", True)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_detect_mps_available_true(self, mock_mps: Mock) -> None:
        result = detect_mps_available()
        assert result is True

    @patch("src.env_detector.TORCH_AVAILABLE", False)
    def test_detect_mps_available_false(self) -> None:
        result = detect_mps_available()
        assert result is False


class TestDetectSystemInfo:
    """Tests for detect_system_info function."""

    @patch("src.env_detector.psutil.virtual_memory")
    @patch("src.env_detector.psutil.cpu_count")
    @patch("src.env_detector.TORCH_AVAILABLE", False)
    @patch("src.env_detector.platform.system", return_value="Darwin")
    def test_detect_system_info_mac(
        self,
        mock_system: Mock,
        mock_cpu_count: Mock,
        mock_virtual_memory: Mock,
    ) -> None:
        mock_cpu_count.return_value = 10
        mock_mem = Mock(total=16 * 1024**3)  # 16 GB
        mock_virtual_memory.return_value = mock_mem

        env, os_name, has_cuda, has_mps, gpu_name, gpu_vram_mb, cpu_count, total_ram_gb = (
            detect_system_info()
        )

        assert env == "mac"
        assert os_name == "darwin"
        assert has_cuda is False
        assert has_mps is False
        assert gpu_name is None
        assert gpu_vram_mb is None
        assert cpu_count == 10
        assert total_ram_gb == 16.0

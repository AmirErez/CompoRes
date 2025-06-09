import tempfile
import os
from src.synthetic_data.analyze_binary_classification.process_auroc_results import parse_auroc_from_file


class TestParseAUROCFromFile:
    def setup_method(self):
        """Set up resources before each test method."""
        self.temp_files = []

    def teardown_method(self):
        """Clean up resources after each test method."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def create_temp_file(self, content):
        """Utility to create a temporary file with given content."""
        temp_file = tempfile.NamedTemporaryFile("w+", delete=False)
        self.temp_files.append(temp_file.name)
        temp_file.write(content)
        temp_file.seek(0)  # Reset the file pointer to the start
        temp_file.close()
        return temp_file.name

    def test_parse_valid_auroc(self):
        """Test parsing a valid AUROC value."""
        file_path = self.create_temp_file("AUC: 0.87\n")
        result = parse_auroc_from_file(file_path)
        assert result == 0.87, f"Expected 0.87 but got {result}"

    def test_parse_auroc_with_crlf(self):
        """Test parsing with Windows-style CRLF line endings."""
        file_path = self.create_temp_file("AUC: 0.87\r\n")
        result = parse_auroc_from_file(file_path)
        assert result == 0.87, f"Expected 0.87 but got {result}"

    def test_parse_auroc_missing_value(self):
        """Test when the file does not contain an AUROC value."""
        file_path = self.create_temp_file("Some other content\n")
        result = parse_auroc_from_file(file_path)
        assert result is None, f"Expected None but got {result}"

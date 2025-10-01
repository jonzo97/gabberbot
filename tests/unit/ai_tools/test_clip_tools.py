#!/usr/bin/env python3
"""
Unit tests for AI Clip Tools

Comprehensive testing of unified clip tool system including:
- MIDI clip creation tools
- Trigger clip creation tools
- Tool parameter validation
- Error handling
- Integration with synthesizers
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cli_shared.ai.clip_tools import (
    CreateMIDIClipTool, CreateTriggerClipTool, SaveClipTool,
    LoadClipTool, ExportClipTool, ManipulateClipTool, create_clip_tools
)
from cli_shared.models.midi_clips import MIDIClip, TriggerClip
from cli_shared.interfaces.synthesizer import MockSynthesizer


class TestCreateMIDIClipTool(unittest.TestCase):
    """Test CreateMIDIClipTool class"""
    
    def setUp(self):
        """Set up test tool"""
        self.synthesizer = MockSynthesizer()
        self.tool = CreateMIDIClipTool(self.synthesizer)
    
    def test_tool_initialization(self):
        """Test tool creation"""
        self.assertEqual(self.tool.name, "create_midi_clip")
        self.assertIn("Create melodic MIDI clips", self.tool.description)
        self.assertEqual(self.tool.synthesizer, self.synthesizer)
    
    async def test_acid_bassline_creation(self):
        """Test acid bassline generation"""
        result = await self.tool.execute(
            clip_type="acid_bassline",
            length_bars=4.0,
            bpm=180.0,
            scale="A_minor",
            accent_pattern="x ~ ~ x ~ ~ x ~"
        )
        
        # Check result structure (ClipToolResult)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.clip)
        self.assertIsNotNone(result.data)
        
        # Check generated clip
        clip = result.clip
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 4.0)
        self.assertEqual(clip.bpm, 180.0)
        self.assertEqual(clip.key_signature, "A_minor")
        self.assertIn("acid", clip.tags)
        
        # Check synthesizer interaction (MockSynthesizer doesn't track clips the same way)
        self.assertIsNotNone(result.tidal_pattern)
    
    async def test_tuned_kick_creation(self):
        """Test tuned kick generation"""
        result = await self.tool.execute(
            clip_type="tuned_kick",
            length_bars=2.0,
            bpm=200.0,
            root_note="C2",
            pattern="x ~ x ~ x ~ x x",
            tuning="pentatonic"
        )
        
        # Check result
        self.assertTrue(result.success)
        clip = result.clip
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 2.0)
        self.assertEqual(clip.bpm, 200.0)
        self.assertIn("tuned_kick", clip.tags)
        self.assertIn("frenchcore", clip.tags)
    
    async def test_riff_creation_fallback(self):
        """Test riff creation (should use acid bassline as fallback)"""
        result = await self.tool.execute(
            clip_type="riff",
            length_bars=8.0,
            bpm=140.0,
            style="industrial"
        )
        
        # Should succeed using fallback
        self.assertTrue(result.success)
        clip = result.clip
        self.assertIsInstance(clip, MIDIClip)
        self.assertEqual(clip.length_bars, 8.0)
    
    async def test_invalid_clip_type(self):
        """Test handling of invalid clip type"""
        result = await self.tool.execute(
            clip_type="invalid_type",
            length_bars=4.0,
            bpm=180.0
        )
        
        # Should fail for invalid type
        self.assertFalse(result.success)
        self.assertIn("Unknown clip type", result.error)
    
    async def test_parameter_validation(self):
        """Test parameter validation and defaults"""
        # Test with minimal parameters
        result = await self.tool.execute(clip_type="acid_bassline")
        
        self.assertTrue(result.success)
        clip = result.clip
        self.assertEqual(clip.length_bars, 4.0)  # Default
        self.assertEqual(clip.bpm, 180.0)  # Default
        
        # Test with extreme parameters
        result = await self.tool.execute(
            clip_type="acid_bassline",
            length_bars=100.0,  # Very long
            bpm=300.0  # Very fast
        )
        
        self.assertTrue(result.success)
        clip = result.clip
        self.assertEqual(clip.length_bars, 100.0)
        self.assertEqual(clip.bpm, 300.0)
    
    async def test_synthesizer_error_handling(self):
        """Test handling of synthesizer errors"""
        # Create mock that raises exception
        failing_synthesizer = Mock(spec=AbstractSynthesizer)
        failing_synthesizer.synthesize_clip = AsyncMock(side_effect=Exception("Synthesis failed"))
        
        failing_tool = CreateMIDIClipTool(failing_synthesizer)
        
        result = await failing_tool.execute(clip_type="acid_bassline")
        
        # Should handle error gracefully
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("Synthesis failed", result.error)


class TestCreateTriggerClipTool(unittest.TestCase):
    """Test CreateTriggerClipTool class"""
    
    def setUp(self):
        """Set up test tool"""
        self.synthesizer = MockSynthesizer()
        self.tool = CreateTriggerClipTool(self.synthesizer)
    
    def test_tool_initialization(self):
        """Test tool creation"""
        self.assertEqual(self.tool.name, "create_trigger_clip")
        self.assertIn("Create trigger-based clips for drums and percussion", self.tool.description)
    
    async def test_kick_pattern_creation(self):
        """Test kick pattern generation"""
        result = await self.tool.execute(
            clip_type="kick_pattern",
            length_bars=1.0,
            bpm=180.0,
            pattern="x ~ x ~ x ~ x ~",
            sample_id="kick_909"
        )
        
        # Check result
        self.assertTrue(result.success)
        clip = result.clip
        self.assertIsInstance(clip, TriggerClip)
        self.assertEqual(clip.length_bars, 1.0)
        self.assertEqual(clip.bpm, 180.0)
        self.assertGreater(len(clip.triggers), 0)
        
        # Check triggers
        for trigger in clip.triggers:
            self.assertEqual(trigger.sample_id, "kick_909")
            self.assertEqual(trigger.channel, 9)  # Drum channel
    
    async def test_complex_gabber_pattern(self):
        """Test complex gabber pattern with multiple samples"""
        result = await self.tool.execute(
            clip_type="gabber_pattern",
            length_bars=2.0,
            bpm=190.0,
            kick_pattern="x x x x x x x x",
            hihat_pattern="~ x ~ x ~ x ~ x",
            snare_pattern="~ ~ x ~ ~ ~ x ~"
        )
        
        # Check result
        self.assertTrue(result.success)
        clip = result.clip
        self.assertIsInstance(clip, TriggerClip)
        self.assertEqual(clip.length_bars, 2.0)
        
        # Should have multiple trigger types
        sample_ids = set(trigger.sample_id for trigger in clip.triggers)
        self.assertIn("kick", sample_ids)
        self.assertIn("hihat", sample_ids)
        self.assertIn("snare", sample_ids)
    
    async def test_industrial_pattern_creation(self):
        """Test industrial pattern generation"""
        result = await self.tool.execute(
            clip_type="industrial_pattern",
            length_bars=4.0,
            bpm=140.0,
            density=0.7,
            reverb_amount=0.8
        )
        
        self.assertTrue(result.success)
        clip = result.clip
        self.assertIn("industrial", clip.tags)
    
    async def test_empty_pattern_handling(self):
        """Test handling of empty patterns"""
        result = await self.tool.execute(
            clip_type="kick_pattern",
            pattern="~ ~ ~ ~",  # All rests
            length_bars=1.0
        )
        
        # Should create empty clip without error
        self.assertTrue(result.success)
        clip = result.clip
        self.assertEqual(len(clip.triggers), 0)


class TestLibraryTools(unittest.TestCase):
    """Test clip library management tools"""
    
    def setUp(self):
        """Set up test tools"""
        self.synthesizer = MockSynthesizer()
        self.save_tool = SaveClipTool(self.synthesizer)
        self.load_tool = LoadClipTool(self.synthesizer)
    
    async def test_library_directory_creation(self):
        """Test that library directories are created"""
        # Both tools should create library directories
        self.assertTrue(hasattr(self.save_tool, 'library_path'))
        self.assertTrue(hasattr(self.load_tool, 'library_path'))
    
    @patch('builtins.open')
    @patch('os.makedirs')
    async def test_save_clip_success(self, mock_makedirs, mock_open):
        """Test successful clip saving"""
        # Create test clip
        clip = MIDIClip(name="test_save", bpm=180.0)
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = await self.save_tool.execute(
            name="test_save",
            clip=clip
        )
        
        # Check result structure matches ClipToolResult
        self.assertTrue(result.success)
        self.assertIn("name", result.data)
        
        # Verify file operations
        mock_open.assert_called_once()
        mock_file.write.assert_called_once()
    
    @patch('builtins.open')
    @patch('os.path.exists')
    async def test_load_clip_success(self, mock_exists, mock_open):
        """Test successful clip loading"""
        mock_exists.return_value = True
        
        # Mock file content
        clip_data = {
            "name": "test_load",
            "length_bars": 4.0,
            "bpm": 180.0,
            "notes": [],
            "time_signature": {"numerator": 4, "denominator": 4},
            "key_signature": "C",
            "created_at": 1234567890.0,
            "tags": ["test"],
            "genre": "hardcore"
        }
        
        mock_file = Mock()
        mock_file.read.return_value = str(clip_data).replace("'", '"')
        mock_open.return_value.__enter__.return_value = mock_file
        
        with patch('json.loads', return_value=clip_data):
            result = await self.load_tool.execute(source="library", name="test_load")
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.clip, MIDIClip)
        self.assertEqual(result.clip.name, "test_load")
    
    @patch('os.path.exists')
    async def test_load_clip_not_found(self, mock_exists):
        """Test loading non-existent clip"""
        mock_exists.return_value = False
        
        result = await self.load_tool.execute(source="library", name="nonexistent")
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)


class TestClipToolsFactory(unittest.TestCase):
    """Test clip tools factory function"""
    
    def test_create_clip_tools(self):
        """Test tool factory function"""
        synthesizer = MockSynthesizer()
        tools = create_clip_tools(synthesizer)
        
        # Check all expected tools are created
        expected_tools = [
            "create_midi_clip",
            "create_trigger_clip", 
            "manipulate_clip",
            "export_clip",
            "load_clip",
            "save_clip"
        ]
        
        for tool_name in expected_tools:
            self.assertIn(tool_name, tools)
            self.assertTrue(hasattr(tools[tool_name], "execute"))
        
        # Check tools have correct synthesizer
        for tool in tools.values():
            self.assertEqual(tool.synthesizer, synthesizer)
    
    def test_tool_names_and_descriptions(self):
        """Test tool metadata"""
        synthesizer = MockSynthesizer()
        tools = create_clip_tools(synthesizer)
        
        # Check each tool has proper name and description
        for tool_name, tool in tools.items():
            self.assertEqual(tool.name, tool_name)
            self.assertIsInstance(tool.description, str)
            self.assertGreater(len(tool.description), 10)  # Non-trivial description


class TestToolIntegration(unittest.TestCase):
    """Test tool integration scenarios"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.synthesizer = MockSynthesizer()
        self.tools = create_clip_tools(self.synthesizer)
    
    async def test_create_save_load_workflow(self):
        """Test complete workflow: create -> save -> load"""
        # Create clip
        create_result = await self.tools["create_midi_clip"].execute(
            clip_type="acid_bassline",
            length_bars=2.0,
            bpm=160.0
        )
        
        self.assertTrue(create_result.success)
        original_clip = create_result.clip
        
        # Save clip (mock file operations)
        with patch('builtins.open'), patch('os.makedirs'):
            save_result = await self.tools["save_clip"].execute(
                name="test_workflow",
                clip=original_clip
            )
        
        self.assertTrue(save_result.success)
        
        # Load clip (mock file operations)
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open'), \
             patch('json.loads', return_value=original_clip.to_dict()):
            
            load_result = await self.tools["load_clip"].execute(
                source="library",
                name="test_workflow"
            )
        
        self.assertTrue(load_result.success)
        loaded_clip_data = load_result.clip
        
        # Verify data integrity
        self.assertEqual(loaded_clip_data.name, original_clip.name)
        self.assertEqual(loaded_clip_data.bpm, original_clip.bpm)
        self.assertEqual(loaded_clip_data.length_bars, original_clip.length_bars)
    
    async def test_create_multiple_clips(self):
        """Test creating multiple different clip types"""
        clip_specs = [
            {"clip_type": "acid_bassline", "bpm": 180.0},
            {"clip_type": "tuned_kick", "bpm": 200.0, "root_note": "C2"},
        ]
        
        results = []
        for spec in clip_specs:
            result = await self.tools["create_midi_clip"].execute(**spec)
            self.assertTrue(result.success)
            results.append(result)
        
        # Verify different clips were created
        self.assertNotEqual(results[0].clip.tags, results[1].clip.tags)
    
    async def test_trigger_and_midi_clip_creation(self):
        """Test creating both trigger and MIDI clips"""
        # Create MIDI clip
        midi_result = await self.tools["create_midi_clip"].execute(
            clip_type="acid_bassline"
        )
        
        # Create trigger clip
        trigger_result = await self.tools["create_trigger_clip"].execute(
            clip_type="kick_pattern",
            pattern="x ~ x ~ x ~ x ~"
        )
        
        # Both should succeed
        self.assertTrue(midi_result.success)
        self.assertTrue(trigger_result.success)
        
        # Should be different types
        self.assertIsInstance(midi_result.clip, MIDIClip)
        self.assertIsInstance(trigger_result.clip, TriggerClip)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
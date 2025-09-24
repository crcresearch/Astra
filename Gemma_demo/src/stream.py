class BaseStream:
    """Abstract base class for any recording hardware."""
    def start(self):
        """Start the device."""
        raise NotImplementedError("Start method not implemented")

    def stop(self):
        """Stop the device."""
        raise NotImplementedError("Stop method not implemented")

    def read(self):
        """Return the latest data sample (frame, audio buffer, sensor reading, etc.)."""
        raise NotImplementedError("Read method not implemented")
    
    def simulate_read(self):
        """Return a simulated reading sample for testing without actual hardware."""
        raise NotImplementedError("Simulate read method not implemented")
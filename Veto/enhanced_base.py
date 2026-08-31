import threading
import time
from base import EchoForQwen

tls = threading.local()

class ThreadAwareTrackingClient:
    def __init__(self, original_client, tls):
        self._original = original_client
        self.tls = tls
        self.chat = self.Chat(original_client.chat, tls)

    class Chat:
        def __init__(self, original_chat, tls):
            self.completions = self.Completions(original_chat.completions, tls)

        class Completions:
            def __init__(self, original_completions, tls):
                self._original_completions = original_completions
                self.tls = tls

            def create(self, *args, **kwargs):
                start_time = time.time()
                res = self._original_completions.create(*args, **kwargs)
                elapsed = time.time() - start_time
                
                tokens = 0
                if hasattr(res, 'usage') and res.usage:
                    tokens = getattr(res.usage, 'total_tokens', 0)
                
                state = getattr(self.tls, 'state', None)
                if state is not None and hasattr(state, 'tracker'):
                    with state.tracker['lock']:
                        state.tracker['total_tokens'] += tokens
                        state.tracker['total_api_time'] += elapsed
                
                return res

class EnhancedEchoForQwen(EchoForQwen):
    def __init__(self, config=None):
        super().__init__(config)
        # Override the client with our tracking wrapper
        self.client = ThreadAwareTrackingClient(self.client, tls)

    def find_targets(self, state):
        # Initialize tracker in state
        if not hasattr(state, 'tracker'):
            state.tracker = {
                'total_tokens': 0,
                'total_api_time': 0.0,
                'lock': threading.Lock()
            }
        
        # Set thread-local state
        tls.state = state
        super().find_targets(state)

    def ground_target_instances(self, state, target):
        # This method is called in a ThreadPoolExecutor sub-thread.
        # We need to set the thread-local state for this sub-thread.
        tls.state = state
        return super().ground_target_instances(state, target)

    def reasoning_step(self, state):
        # Ensure thread-local state is set (in case it was cleared or running in a different context)
        tls.state = state
        return super().reasoning_step(state)
        
    def remove_unnecessary(self, state, mode="efficient"):
        tls.state = state
        return super().remove_unnecessary(state, mode)

    def generate_with_tracking(self, image_pil, question, custom_debug_dir=None):
        tls.state = None
        start_time = time.time()
        
        # Call the original generate method from EchoForQwen
        # The result will be computed and our overridden methods will handle tracking hooks
        result = super().generate(image_pil, question, custom_debug_dir)
        
        total_wall_time = time.time() - start_time
        
        tracker_data = {
            'total_tokens': 0,
            'total_api_time': 0.0,
            'total_wall_time': total_wall_time
        }
        
        if getattr(tls, 'state', None) and hasattr(tls.state, 'tracker'):
            tracker_data['total_tokens'] = tls.state.tracker['total_tokens']
            tracker_data['total_api_time'] = tls.state.tracker['total_api_time']
            # Clear tls.state to prevent memory leaks
            tls.state = None
            
        return result, tracker_data

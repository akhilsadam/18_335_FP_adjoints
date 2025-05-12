import sys, types, logging, inspect, time
import torch

# TODO move memory and gradient magnitude debug scripts in from debug.py

logger = logging.getLogger('Anzen')

def get_tensors(d):
    u = {}
    for k,v in d.items():
        if torch.is_tensor(v):
            u[(k,v.data_ptr())] = v
    return u


class AnzenTorchTracer():
    def __init__(self, filenames, 
                 fwd_checks, bwd_checks, 
                 fwd_timeout=-2):
        self.filenames = filenames
    
        self._reset()
        self.set_forward_trace()
        
        self.fwd_checks = fwd_checks
        self.bwd_checks = bwd_checks
        
        self.fwd_time_checks = fwd_timeout > -1
        if self.fwd_time_checks:
            self.fwd_frame_times = {}
            self.fwd_code_times = {}
            self.timeout = fwd_timeout
        
    def _reset(self):
        self.tensors = {}
        self.tensor_grads = {}
        
        self.forward_stack = []
        self.location_stack = []
        self.backward_stack = []
        self.stack_length = 0

    def step(self):
        self._reset()
        
    ####################### Backward tracer
    def backward(self):

        for i in range(self.stack_length):
            j = self.stack_length - i - 1
            new_grad_keys = [k for k in self.forward_stack[j] if self.tensors[k].requires_grad and k in self.tensor_grads.keys()]
            loc = self.location_stack[j]

            for check in self.bwd_checks:
                check(self.tensors,self.tensor_grads,new_grad_keys,loc,j)

    def add_hooks(self, new_tensors):
        for k,v in new_tensors.items():
            if v.requires_grad:
                def hook(grad):
                    # print(f"hook {k}")
                    self.tensor_grads[k] = grad.detach().clone()
                v.register_hook(hook)
    ####################### Forward tracer
    def add_to_fwd_stack(self, loc, new_keys, new_tensors):
        i = self.stack_length
        logger.info(f"FWD {i}:{loc} - new tensors: {new_keys}")#[k[0] for k in new_keys]
        self.forward_stack.append(new_keys)
        self.location_stack.append(loc)
        self.add_hooks(new_tensors)
        self.tensors.update(new_tensors)
        self.stack_length += 1
    
    def forward_step(self, frame: types.FrameType, loc: tuple):
            new_tensors = get_tensors(frame.f_locals)
            new_keys = [k for k in new_tensors.keys() if k not in self.tensors.keys()]
            if len(new_keys) > 0:
                self.add_to_fwd_stack(loc, new_keys, new_tensors)
            
                for check in self.fwd_checks:
                    check(self.tensors,new_keys,loc,self.stack_length)

                # try:
                #     print(inspect.getsourcelines(frame))
                # except:
                #     pass
        
    def _fwd_call(self, frame: types.FrameType, event, arg):
        if event == 'call':# or event == 'call' or event == 'return':
            # time checks
            if self.fwd_time_checks:
                self.fwd_frame_times[frame] = time.time()
            
            code = frame.f_code
            filename = code.co_filename.split('/')[-1]
            if code.co_name not in ['<module>'] and filename in self.filenames:
                self.forward_step(frame,loc=(filename,code.co_firstlineno))
                return self._fwd_lines
                   
        elif event == 'return':
            if self.fwd_time_checks:
                start_time = self.fwd_frame_times[frame]
                elapsed_time = time.time() - start_time
                self.fwd_code_times[frame.f_code.co_name] = elapsed_time
                if elapsed_time > self.timeout:
                    logger.warn(f"Code {frame.f_code.co_name} took {elapsed_time} seconds")
                     
        elif event == 'exception':
            pass
                # self._handle_return(frame)
    def _fwd_lines(self, frame: types.FrameType, event, arg):
        if event == 'line':# or event == 'call' or event == 'return':
            code = frame.f_code
            filename = code.co_filename.split('/')[-1]
            self.forward_step(frame,loc=(filename,code.co_name,frame.f_lineno))

    def set_forward_trace(self):
        sys.settrace(self._fwd_call)
    def unset_forward_trace(self):
        sys.settrace(None)                  


class AnzenSuite:
    
    def nan_forward(tensors,new_keys,loc,i):
        for i in new_keys:
            g = tensors[i]
            if g is not None:
                assert torch.isfinite(g).all(), ValueError(f'Tensor {i} has nan value at {loc} from step {i}')
                
    def maxmin_forward(tensors,new_keys,loc,j):
        info = []
        # print(tensor_grads.keys())
        for i in new_keys:
            g = tensors[i]
            if g is not None:
                info.append((i[0],torch.min(torch.abs(g)),torch.max(torch.abs(g))))
        out = [f"{i[0]}:{i[1]} to {i[2]}" for i in info]
        logger.info(f'step {j} {loc}: {out}')
        
    def maxmin_gradient(tensors,tensor_grads,new_grad_keys,loc,j):
        info = []
        # print(tensor_grads.keys())
        for gi in new_grad_keys:
            g = tensor_grads[gi]
            if g is not None:
                info.append((gi[0],torch.min(torch.abs(g)),torch.max(torch.abs(g))))
        out = [f"{i[0]}:{i[1]} to {i[2]}" for i in info]
        logger.info(f'step {j} {loc}: {out}')
    
    def nan_gradient(tensors,tensor_grads,new_grad_keys,loc,i):
        for gi in new_grad_keys:
            g = tensor_grads[gi]
            if g is not None:
                assert torch.isfinite(g).all(), ValueError(f'Tensor {gi} has nan gradient at {loc} from step {i}')


# To use
# from anzen_torch import AnzenTorchTracer as att
# sys.settrace(att(<filenames>).dispatch_trace)
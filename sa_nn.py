'''
COPYRIGHT(C),AFIF ALI SAADMAN, 21st Century and BEYOND
This software/python module DOES NOT COME WITH ANY WARRANTY.
This python module can run Neural Networks in an Arduino with the `use_progmem` param being set to true.
'''
import torch
import torch.nn as nn
import inspect
import ast
from typing import Optional, Dict, List
import abc


def detect_activations_in_forward(model):
    """
    Detect activation functions used in the forward method by analyzing the source code.
    INTERNAL FUNCTION. NOT TO BE USED BY A NORMAL USER.
    """
    try:
        source = inspect.getsource(model.forward);tree = ast.parse(source)
        activations = []

        activation_functions = {
            'torch.relu', 'F.relu', 'torch.nn.functional.relu',
            'torch.sigmoid', 'F.sigmoid', 'torch.nn.functional.sigmoid',
            'torch.tanh', 'F.tanh', 'torch.nn.functional.tanh',
            'torch.nn.functional.gelu', 'F.gelu', 'gelu',
            'torch.nn.functional.leaky_relu', 'F.leaky_relu', 'leaky_relu'
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name and any(fn.endswith(func_name) for fn in activation_functions):
                    if 'relu' in func_name.lower(): activations.append(nn.ReLU())
                    elif 'sigmoid' in func_name.lower(): activations.append(nn.Sigmoid())
                    elif 'tanh' in func_name.lower(): activations.append(nn.Tanh())
                    elif 'gelu' in func_name.lower(): activations.append(nn.GELU())
                    elif 'leaky_relu' in func_name.lower(): activations.append(nn.LeakyReLU())
        return activations
    except Exception:
        raise ValueError("AUTOPARSE ERROR: ACTIVATION NOT FOUND INSIDE THE SET : 'relu6','relu','gelu','sigmoid','tanh','leaky_relu'");return []
    


def export_sa_nn(
    model: nn.Module,
    filename: str = "sa_nn_model.h",
    vocab: Optional[Dict[str,int]] = None,
    use_progmem: bool = False  
):
    """
    Export a PyTorch model to embedded-safe C/C++ code (SA-NN) including optional text-to-feature conversion if vocab (a python dictionary) is provided.
    Set use_progmem to true if you want less RAM usage and more usage of flash memory.(arduino only)
    """

    if use_progmem:
        print("BULDING FOR ARCH:ARDUINO (AVRDUDE)")
    layer_stack = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_stack.append(('linear', module))
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.LazyConv1d) or isinstance(module, nn.LazyConv2d) or isinstance(module, nn.LazyConv3d):
            raise RuntimeError("ERROR: Conventional Network detected ... Aborting")
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ReLU6)):
            layer_stack.append(('activation', module))

    layers = []
    layer_to_activation_map = {}
    activations = []

    if len(layers) > 19:
        raise RuntimeError("ERROR: LAYERS COUNT  ARE GREATER THAN 19.")
    for item_type, module in layer_stack:
        if item_type == 'linear':
            layers.append(module)
            layer_to_activation_map[len(layers)-1] = None
        elif item_type == 'activation':
            if len(layers) > 0:
                name = type(module).__name__.lower()
                act_name = 'unknown'
                if 'relu6' in name: act_name='relu6'
                elif 'relu' in name: act_name='relu'
                elif 'gelu' in name: act_name='gelu'
                elif 'sigmoid' in name: act_name='sigmoid'
                elif 'tanh' in name: act_name='tanh'
                elif 'leakyrelu' in name: act_name='leaky_relu'
                else:
                    raise ValueError("AUTOPARSE ERROR: ACTIVATION NOT FOUND INSIDE THE SET : 'relu6','relu','gelu','sigmoid','tanh','leaky_relu'")
                layer_to_activation_map[len(layers)-1] = act_name
            activations.append(module)

    if not layers:
        raise ValueError("Model must contain at least one Linear layer")

    signs, shifts, biases = [], [], []
    for layer in layers:
        weight = layer.weight.data
        bias = layer.bias.data

        sign_weights = torch.sign(weight).to(torch.int8)
        abs_weights = torch.abs(weight)
        shift_values = torch.clamp(torch.floor(torch.log2(abs_weights+1e-8)), 0, 7).to(torch.uint8)
        signs.append(sign_weights);shifts.append(shift_values);biases.append(bias.to(torch.int16))

    if vocab:
        c_code = conversionDaemon(layer_to_activation_map,layers,signs,shifts,biases,use_progmem,vocab)
    else:
        c_code = conversionDaemon(layer_to_activation_map,layers,signs,shifts,biases,use_progmem)

    

    with open(filename,"w") as f:
        f.write("\n".join(c_code))


def conversionDaemon(layer_to_activation_map : dict ,layers : int, signs : list, shifts : list, biases : list, use_progmem : bool = False, vocab : dict = None):
    print("\n \n \n --------------SA-NN CONVERSION DAEMON ACTIVATED revision : 1--------------------")
    if use_progmem:
        print("\n \n \n --------------USING ARCH AS HEADER TARGER -> ARDUINO (AVR) --------------")
    c_code = []
    c_code.append("// SA-NN Interference Header Generated by SA-NN ( SPELLING MISTAKE AINT BEING REPAIRED)")
    c_code.append("#include <stdint.h>")
    if use_progmem:
        c_code.append("#include <avr/pgmspace.h>")
    c_code.append("#ifndef SA_NN_MODEL_H")
    c_code.append("#define SA_NN_MODEL_H\n")

    # layer defines
    for i, layer in enumerate(layers):
        c_code.append(f"#define L{i}_IN {layer.in_features}")
        c_code.append(f"#define L{i}_OUT {layer.out_features}")
    c_code.append("")

    for i, s in enumerate(signs):
        if use_progmem:
            c_code.append(f"const int8_t SIGN_L{i}[{s.shape[0]}][{s.shape[1]}] PROGMEM = {{")
        else:
            c_code.append(f"const int8_t SIGN_L{i}[{s.shape[0]}][{s.shape[1]}] = {{")
        for j,row in enumerate(s):
            c_code.append("  {" + ", ".join(str(int(v)) for v in row) + "}" + ("," if j < s.shape[0]-1 else ""))
        c_code.append("};")
    c_code.append("")

    for i, sh in enumerate(shifts):
        if use_progmem:
            c_code.append(f"const uint8_t SHIFT_L{i}[{sh.shape[0]}][{sh.shape[1]}] PROGMEM = {{")
        else:
            c_code.append(f"const uint8_t SHIFT_L{i}[{sh.shape[0]}][{sh.shape[1]}] = {{")
        for j,row in enumerate(sh):
            c_code.append("  {" + ", ".join(str(int(v)) for v in row) + "}" + ("," if j < sh.shape[0]-1 else ""))
        c_code.append("};")
    c_code.append("")

    # bias arrays
    for i, b in enumerate(biases):
        if use_progmem:
            c_code.append(f"const int16_t BIAS_L{i}[{len(b)}] PROGMEM = {{ " + ", ".join(str(int(v)) for v in b) + " };")
        else:
            c_code.append(f"const int16_t BIAS_L{i}[{len(b)}] = {{ " + ", ".join(str(int(v)) for v in b) + " };")
    c_code.append("")

    c_code.append("#define MAX_VAL 32764\n") #  to reduce OVERFLOW

    # activation functions inline
    act_used = set([v for v in layer_to_activation_map.values() if v])
    activationDefines(act_used,c_code)

    func_name = "interfere"
    if vocab and isinstance(vocab, dict) and len(vocab)>0:
        func_name = "interfere_text"
        
        vocab_items = sorted(vocab.items(), key=lambda x:x[1])
        c_code.append(f"const int VOCAB_SIZE={len(vocab)};")
        for word, idx in vocab_items:
            escaped = word.replace('"','\\"')
            if use_progmem:
                c_code.append(f'const char VOCAB_WORD_{idx}[] PROGMEM = "{escaped}";')
            else:
                c_code.append(f'const char VOCAB_WORD_{idx}[] = "{escaped}";')
        
        if use_progmem:
            c_code.append("""
int string_equals(const char* s1,const char* s2_pgm,int max_len){
    for(int i=0;i<max_len;i++){
        char s2_char = pgm_read_byte(&(s2_pgm[i]));
        if(s1[i]!=s2_char) return 0;
        if(s1[i]=='\\0' && s2_char=='\\0') return 1;
        if(s1[i]=='\\0' || s2_char=='\\0') return 0;
    }
    char s2_last = pgm_read_byte(&(s2_pgm[max_len]));
    return (s1[max_len]=='\\0' && s2_last=='\\0');
}
""")
        else:
            c_code.append("""
int string_equals(const char* s1,const char* s2,int max_len){
    for(int i=0;i<max_len;i++){
        if(s1[i]!=s2[i]) return 0;
        if(s1[i]=='\\0') return 1;
    }
    return (s1[max_len]=='\\0' && s2[max_len]=='\\0');
}
""")
        # find_vocab_index
        c_code.append("int find_vocab_index(const char* word,int word_len){")
        for word, idx in vocab_items:
            c_code.append(f"  if(string_equals(word,VOCAB_WORD_{idx},word_len)) return {idx};")
        c_code.append("  return -1;\n}")
        # string_to_features
        c_code.append("""
void string_to_features(const char* text,int8_t* features,int max_len){
    for(int i=0;i<max_len && i<VOCAB_SIZE;i++) features[i]=0;
    int text_len=0; while(text[text_len]!='\\0') text_len++;
    int start=0;
    for(int i=0;i<=text_len;i++){
        if(text[i]==' ' || text[i]=='\\0'){
            if(i>start){
                char token[50]; int token_len=i-start;
                if(token_len>=50) token_len=49;
                for(int k=0;k<token_len;k++) token[k]=text[start+k];
                token[token_len]='\\0';
                for(int k=0;k<token_len;k++)
                    if(token[k]>='A' && token[k]<='Z') token[k]+=('a'-'A');
                int vid=find_vocab_index(token,token_len);
                if(vid>=0 && vid<max_len) features[vid]=1;
            }
            start=i+1;
        }
    }
}
""")

    input_var = "processed_input" if func_name=="interfere_text" else "input"
    if func_name=="interfere_text":
        c_code.append(f"void {func_name}(const char* text,int16_t* output){{int8_t processed_input[VOCAB_SIZE];string_to_features(text,processed_input,VOCAB_SIZE);")
    else:
        c_code.append(f"void {func_name}(const int8_t* input,int16_t* output){{")

    for i, _ in enumerate(layers):
        c_code.append(f"  int16_t layer_{i}[L{i}_OUT];")

    for i, layer in enumerate(layers):
        c_code.append(f"  for(int j=0;j<L{i}_OUT;j++){{")
        if use_progmem:
            c_code.append(f"    int32_t sum=pgm_read_word(&(BIAS_L{i}[j]));")
        else:
            c_code.append(f"    int32_t sum=BIAS_L{i}[j];")
        c_code.append(f"    for(int k=0;k<L{i}_IN;k++){{")
        if i==0:
            c_code.append(f"      int16_t val={input_var}[k];")
        else:
            c_code.append(f"      int16_t val=layer_{i-1}[k];")
        if use_progmem:
            c_code.append(f"      sum+=(int8_t)pgm_read_byte(&(SIGN_L{i}[j][k]))*(val<<(uint8_t)pgm_read_byte(&(SHIFT_L{i}[j][k])));")
        else:
            c_code.append(f"      sum+=SIGN_L{i}[j][k]*(val<<SHIFT_L{i}[j][k]);")
        c_code.append("    }")
        act = layer_to_activation_map.get(i)
        if act:
            c_code.append(f"    sum={act}((int16_t)sum);")
        c_code.append("    if(sum<0)sum=0;else if(sum>MAX_VAL)sum=MAX_VAL;")
        if i==len(layers)-1:
            c_code.append("    output[j]=(int16_t)sum;")
        else:
            c_code.append(f"    layer_{i}[j]=(int16_t)sum;")
        c_code.append("  }")
    c_code.append("}")
    c_code.append("#endif // SA_NN_MODEL_H")
    return c_code

def activationDefines(act_used : set,c_code : list) :
    if 'relu' in act_used:
        c_code.append("int16_t relu(int16_t x){return x>0?x:0;}")
    if 'leaky_relu' in act_used:
        c_code.append("int16_t leaky_relu(int16_t x){return x>0?x:(x/100);}")  # set slope to 0.01 (alpha value) 
    if 'sigmoid' in act_used:
        # sigmoid function piecewise approximation
        c_code.append("int16_t sigmoid(int16_t x){")
        c_code.append("    if(x < -4096) return 500;    // ~0.015 scaled to int16")
        c_code.append("    if(x > 4096) return 32267;   // ~0.985 scaled to int16")
        c_code.append("")
        c_code.append("    int32_t temp = (int32_t)x * 4;  // steeeeeeeeeepr slope near center")
        c_code.append("    return (int16_t)(16384 + temp);")
        c_code.append("}")
    if 'tanh' in act_used:
        c_code.append("int16_t tanh_act(int16_t x){")
        c_code.append("    if(x < -4096) return -32000;  // close to -1")
        c_code.append("    if(x > 4096) return 32000;   // close to +1")
        c_code.append("    //  approxing with steeper slope")
        c_code.append("    return (int16_t)x;  ")
        c_code.append("}")
    if 'gelu' in act_used:
        #  gelu approximation - defining sigmoid once for use in gelu
        c_code.append("int16_t gelu(int16_t x){")
        c_code.append("    // approximation: x * phi(x) ≈ 0.5 * x * (1 + tanh(0.797885 * (x + 0.044715 * x^3)))")
        c_code.append("    int32_t x_cube = ((int32_t)x * x * x) / 32768; ") # For overfliow
        c_code.append("    int32_t inner = x + (x_cube * 147) / 3277;  // 0.044715 * 32768")
        c_code.append("    inner = (inner * 2621) / 3277;  // 0.797885 * 32768/10")
        c_code.append("    int16_t tanh_approx;")
        c_code.append("    if(inner < -4096) tanh_approx = -32000;")
        c_code.append("    else if(inner > 4096) tanh_approx = 32000;")
        c_code.append("    else tanh_approx = (int16_t)inner;")
        c_code.append("    int32_t result = ((int32_t)x * (32768 + tanh_approx)) / 65536;  // division by 2")
        c_code.append("    return (int16_t)result;")
        c_code.append("}")
        c_code.append(" ")
    

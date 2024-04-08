from . import functional,LIFnode
from .LIFAct import LIFAct,LIFAct_withoutbn
from .node_compute_spike_rate import *
node_dict={
    "LIFnode":LIFnode.MultiStepLIFNode,
    "LIFnode_csr":LIFnode_csr,
    "LIFnode_csr_t":LIFnode_csr_t,
    "LIFAct_MPBN":LIFAct,
    "LIFAct":LIFAct_withoutbn,
    "LIFAct_csr":LIFAct_csr,
    "LIFAct_csr_t":LIFAct_csr_t,
}
def init_node(node):
    return node_dict[node]
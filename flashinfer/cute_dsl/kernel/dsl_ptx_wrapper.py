from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int, Boolean, Int32, Float32, Numeric, as_numeric, UInt32

# @dsl_user_op
# def exp2(a: Union[float, Float32], *, loc=None, ip=None) -> Float32:
#     return Float32(
#         llvm.inline_asm(
#             T.f32(),
#             [Float32(a).ir_value(loc=loc, ip=ip)],
#             "ex2.approx.ftz.f32 $0, $1;",
#             "=f,f",
#             has_side_effects=True,
#             is_align_stack=False,
#             asm_dialect=llvm.AsmDialect.AD_ATT,
#         )
#     )

@dsl_user_op
def atomic_add(input: cute.Tensor, a: cutlass.Int32) -> cutlass.Int32:
    """
    Perform an atomic addition on the input tensor using NVVM.
    This function assumes that the input tensor is a pointer to an integer type.
    """
    llvm_ptr = input.iterator.llvm_ptr
    res = nvvm.atomicrmw(res=T.i32(), op=nvvm.AtomicOpKind.ADD, ptr=llvm_ptr, a=cutlass.Int32(a).ir_value())
    return res


@dsl_user_op
def st_flag_volatile(flag_addr: UInt32, flag: UInt32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [UInt32(flag_addr).ir_value(loc=loc, ip=ip), UInt32(flag).ir_value(loc=loc, ip=ip)],
        "st.volatile.global.u32 [%1], %0;",
        "r, l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

# __forceinline__ __device__ void st_flag_volatile(uint32_t *flag_addr, uint32_t flag) {
#   asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
# }

@dsl_user_op
def ld_flag_volatile(flag_addr: UInt32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        T.ui32(),
        [UInt32(flag_addr).ir_value(loc=loc, ip=ip)],
        "ld.volatile.global.u32 %0, [%1];",
        "=r, l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

# __forceinline__ __device__ uint32_t ld_flag_volatile(uint32_t *flag_addr) {
#   uint32_t flag;
#   asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#   return flag;
# }

@dsl_user_op
def ld_flag_acquire(flag_addr: UInt32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        T.ui32(),
        [UInt32(flag_addr).ir_value(loc=loc, ip=ip)],
        "ld.acquire.sys.global.u32 %0, [%1];",
        "=r, l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

# __forceinline__ __device__ uint32_t ld_flag_acquire(uint32_t *flag_addr) {
#   uint32_t flag;
#   asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#   return flag;
# }

@dsl_user_op
def st_flag_volatile(flag_addr: UInt32, flag: UInt32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [UInt32(flag_addr).ir_value(loc=loc, ip=ip), UInt32(flag).ir_value(loc=loc, ip=ip)],
        "st.release.sys.global.u32 [%1], %0;",
        "r, l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

# __forceinline__ __device__ void st_flag_release(uint32_t *flag_addr, uint32_t flag) {
#   asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
# }

@dsl_user_op
def add_flag_release(addr: UInt32, value: UInt32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        T.ui32(),
        [UInt32(addr).ir_value(loc=loc, ip=ip), UInt32(value).ir_value(loc=loc, ip=ip)],
        "atom.release.sys.global.add.u32 %0, [%1], %2;",
        "=r, l, r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

# __forceinline__ __device__ uint32_t add_flag_release(uint32_t *addr, uint32_t val) {
#   uint32_t flag;
#   asm volatile("atom.release.sys.global.add.u32 %0, [%1], %2;" : "=r"(flag) : "l"(addr), "r"(val));
#   return flag;
# }
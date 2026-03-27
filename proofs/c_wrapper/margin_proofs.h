#ifndef MARGIN_PROOFS_H
#define MARGIN_PROOFS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FFI_EXPORT __attribute__((visibility("default")))

FFI_EXPORT void margin_proofs_init(void);

FFI_EXPORT int64_t c_trade_reward(int64_t b, int64_t p, int64_t pr, int64_t q, int64_t pp);

FFI_EXPORT int64_t c_trade_balance(int64_t b, int64_t p, int64_t pr, int64_t q);

FFI_EXPORT int64_t c_trade_position(int64_t b, int64_t p, int64_t pr, int64_t q);

#ifdef __cplusplus
}
#endif

#endif
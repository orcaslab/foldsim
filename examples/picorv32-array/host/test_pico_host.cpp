// test_pico_host: pico_host's protocol against a mock kernel, without Verilator or xsim.
//
// MockKernel models foldsim_ctrl's register block as the host sees it (foldsim_regs.h): ID and VERSION, a command
// queue of `queue` entries applied one per target cycle (STATUS FULL while it is full), a reset of its count of
// cycles, DCR writes and reads (a read's push clears STATUS RSP, its response sets it), START (busy until the
// target is), STATUS BUSY over everything queued or in flight, TCYCLES / HCYCLES. Every register access advances it
// `tick` target cycles, as the in-process emulator does. Its target is pico_v80's CSR map with a stand-in for the
// cores: the writes land in a RAM image per tile; after go it runs `run_cycles` cycles, then answers the reads with
// the results it was given. The tests: a run that matches the golden passes and loads every word in place; a wrong
// result fails; a full queue is waited out; a target that stops advancing, or never finishes, is stopped; a kernel
// that is not a foldsim simulator is refused.
#include "pico_host.h"

#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

struct MockKernel : pico::Transport {
  // settings
  uint32_t id = FOLDSIM_ID, queue = 16, tick = 3, run_cycles = 1000;
  bool frozen_after_go = false, never_finishes = false;
  std::map<std::string, uint32_t> results;
  // state
  struct Cmd { uint32_t op, addr, data; };
  std::deque<Cmd> q;
  uint32_t cmd_addr = 0, cmd_data = 0, reset_left = 0, rsp = 0, run_left = 0;
  bool rsp_valid = false, running = false, start_pending = false, ran = false;
  uint64_t tcycles = 0, hcycles = 0, ctrl = 0;
  std::map<uint32_t, uint32_t> ram;   // tile << 10 | word -> data
  std::vector<std::pair<uint32_t, uint32_t>> writes;
  uint64_t accesses = 0;

  int init() override { return 0; }
  bool runs_on_its_own() const override { return false; }

  void cycle() {
    hcycles += 2;
    if (frozen_after_go && ran)
      return;   // the target no longer advances
    ++tcycles;
    if (running && !never_finishes && --run_left == 0)
      running = false;
    if (reset_left) {
      --reset_left;
      return;
    }
    if (q.empty())
      return;
    Cmd c = q.front();
    q.pop_front();
    switch (c.op) {
      case FOLDSIM_CMD_RESET: reset_left = c.data; running = false; ram.clear(); break;
      case FOLDSIM_CMD_DCR_WRITE:
        if (!running && !ran) { ram[c.addr] = c.data; writes.push_back({c.addr, c.data}); }
        break;
      case FOLDSIM_CMD_DCR_READ: {
        static const std::map<uint32_t, const char*> names = {{0, "checksum"}, {1, "trap"}, {2, "tokens"},
            {3, "cycles"}, {4, "last0"}, {5, "last1"}, {6, "last2"}, {7, "last3"}};
        auto it = names.find(c.addr);
        rsp = (it != names.end() && results.count(it->second)) ? results[it->second] : 0;
        rsp_valid = true;
        break;
      }
      case FOLDSIM_CMD_START: running = true; ran = true; run_left = run_cycles; break;
    }
  }
  void advance() { for (uint32_t i = 0; i < tick; ++i) cycle(); }
  uint32_t status() const {
    const bool busy = !q.empty() || reset_left || running;
    return (busy ? FOLDSIM_STATUS_BUSY : 0) | (q.size() >= queue ? FOLDSIM_STATUS_FULL : 0) |
           (rsp_valid ? FOLDSIM_STATUS_RSP : 0);
  }
  int reg_write(uint32_t off, uint32_t v) override {
    ++accesses;
    advance();
    switch (off) {
      case FOLDSIM_REG_CMD_ADDR: cmd_addr = v; break;
      case FOLDSIM_REG_CMD_DATA: cmd_data = v; break;
      case FOLDSIM_REG_CTRL: ctrl = v; break;
      case FOLDSIM_REG_CMD:
        if (q.size() >= queue) return 0;   // the kernel drops a push to a full queue
        if (v == FOLDSIM_CMD_DCR_READ) rsp_valid = false;
        q.push_back({v, cmd_addr, cmd_data});
        break;
      default: break;
    }
    return 0;
  }
  int reg_read(uint32_t off, uint32_t* v) override {
    ++accesses;
    advance();
    switch (off) {
      case FOLDSIM_REG_ID: *v = id; break;
      case FOLDSIM_REG_VERSION: *v = FOLDSIM_VERSION; break;
      case FOLDSIM_REG_STATUS: *v = status(); break;
      case FOLDSIM_REG_DCR_RSP: *v = rsp; break;
      case FOLDSIM_REG_TCYCLES_LO: *v = uint32_t(tcycles); break;
      case FOLDSIM_REG_TCYCLES_HI: *v = uint32_t(tcycles >> 32); break;
      case FOLDSIM_REG_HCYCLES_LO: *v = uint32_t(hcycles); break;
      case FOLDSIM_REG_HCYCLES_HI: *v = uint32_t(hcycles >> 32); break;
      default: *v = 0; break;
    }
    return 0;
  }
};

int failures = 0;
#define EXPECT(c) do { if (!(c)) { printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

std::map<std::string, uint32_t> golden() {
  return {{"checksum", 0xb1617615}, {"trap", 0xf}, {"tokens", 32}, {"cycles", 1000},
          {"last0", 1}, {"last1", 2}, {"last2", 3}, {"last3", 4}};
}

std::vector<std::pair<uint32_t, uint32_t>> load() {
  std::vector<std::pair<uint32_t, uint32_t>> l;
  for (uint32_t t = 0; t < 4; ++t)
    for (uint32_t w = 0; w < 40; ++w)
      l.push_back({(t << 10) | (w < 20 ? w : 0x200 + w), 0x1000 * t + w * 7});
  return l;
}

int run(MockKernel& k, std::map<std::string, uint32_t>* got, pico::Host* host = nullptr) {
  pico::Host local(k);
  pico::Host& h = host ? *host : local;
  if (h.open() != 0)
    return -1;
  return h.run(load(), got);
}

}  // namespace

int main() {
  {
    printf("pass: the results match, every word loaded in place, in order\n");
    MockKernel k; k.results = golden();
    pico::Host h(k);
    std::map<std::string, uint32_t> got;
    EXPECT(run(k, &got, &h) == 0);
    EXPECT(h.check(got, golden()) == 0);
    EXPECT(k.writes == load());
    EXPECT(k.ram.size() == load().size());
    EXPECT(k.ctrl == 1);
    EXPECT(h.target_cycles >= 1000);
    EXPECT(h.commands == 1 + load().size() + 1 + 8);
  }
  {
    printf("mismatch: a wrong checksum is reported\n");
    MockKernel k; k.results = golden(); k.results["checksum"] ^= 1;
    pico::Host h(k);
    std::map<std::string, uint32_t> got;
    EXPECT(run(k, &got, &h) == 0);
    EXPECT(h.check(got, golden()) == 1);
  }
  {
    printf("backpressure: a two-entry queue is waited out, no command lost\n");
    MockKernel k; k.results = golden(); k.queue = 2; k.tick = 1;
    std::map<std::string, uint32_t> got;
    EXPECT(run(k, &got) == 0);
    EXPECT(k.writes == load());
    EXPECT(got == golden());
  }
  {
    printf("frozen: a target that stops advancing is stopped\n");
    MockKernel k; k.results = golden(); k.frozen_after_go = true;
    pico::Host h(k); h.timeout_s = 0.2;
    std::map<std::string, uint32_t> got;
    EXPECT(run(k, &got, &h) != 0);
    EXPECT(h.error.find("has not advanced") != std::string::npos);
  }
  {
    printf("runaway: a target that never finishes is stopped at max_s\n");
    MockKernel k; k.results = golden(); k.never_finishes = true;
    pico::Host h(k); h.max_s = 0.3;
    std::map<std::string, uint32_t> got;
    EXPECT(run(k, &got, &h) != 0);
    EXPECT(h.error.find("of wall time") != std::string::npos);
  }
  {
    printf("not foldsim: a kernel with another id is refused\n");
    MockKernel k; k.id = 0x12345678;
    pico::Host h(k);
    EXPECT(h.open() != 0);
    EXPECT(h.error.find("not a foldsim simulator") != std::string::npos);
    EXPECT(k.q.empty());
  }
  {
    printf("files: load.hex and golden.txt parse, and a short golden is refused\n");
    std::ofstream("t_load.hex") << "000deadbeef\n4010000002a\n";
    auto l = pico::read_load("t_load.hex");
    EXPECT(l.size() == 2 && l[1].first == 0x401 && l[1].second == 0x2a);
    std::ofstream g("t_golden.txt");
    for (auto& [k, v] : golden()) g << k << "=" << std::hex << v << "\n";
    g.close();
    EXPECT(pico::read_golden("t_golden.txt") == golden());
    std::ofstream("t_short.txt") << "checksum=1\n";
    bool threw = false;
    try { pico::read_golden("t_short.txt"); } catch (const std::exception&) { threw = true; }
    EXPECT(threw);
  }
  printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}

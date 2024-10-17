import sys
from random import randint
import pyray as rl
from fonts import fonts as font_codes


class Chip8:
    def __init__(self):
        self.memory: list[int] = [0] * 4096
        self.registers: list[int] = [0] * 16
        self.I: int = 0
        self.delay_timer_register: int = 0
        self.sound_timer_register: int = 0
        self.PC: int = 0x200
        self.SP: int = 0
        self.stack: list[int] = [0] * 16
        self.display: list[list[int]] = [[0] * 32 for _ in range(64)]
        self.fonts: list[int] = font_codes
        self.tone = None
        self.keyboard_map = {
            rl.KeyboardKey.KEY_ONE.value: 0x1,
            rl.KeyboardKey.KEY_TWO.value: 0x2,
            rl.KeyboardKey.KEY_THREE.value: 0x3,
            rl.KeyboardKey.KEY_FOUR.value: 0xC,
            rl.KeyboardKey.KEY_Q.value: 0x4,
            rl.KeyboardKey.KEY_W.value: 0x5,
            rl.KeyboardKey.KEY_E.value: 0x6,
            rl.KeyboardKey.KEY_R.value: 0xD,
            rl.KeyboardKey.KEY_A.value: 0x7,
            rl.KeyboardKey.KEY_S.value: 0x8,
            rl.KeyboardKey.KEY_D.value: 0x9,
            rl.KeyboardKey.KEY_F.value: 0xE,
            rl.KeyboardKey.KEY_Z.value: 0xA,
            rl.KeyboardKey.KEY_X.value: 0x0,
            rl.KeyboardKey.KEY_C.value: 0xB,
            rl.KeyboardKey.KEY_V.value: 0xF,
        }
        self.is_already_playing = False

    def init_emu(self) -> None:
        # fonts are put in memory
        for i in range(len(self.fonts)):
            self.memory[i] = self.fonts[i]

        rl.init_audio_device()
        self.tone = rl.load_sound("beep.wav")
        rl.set_master_volume(100)

    def close_emu(self) -> None:
        rl.close_audio_device()

    def beep(self) -> None:
        if not self.is_already_playing:
            rl.play_sound(self.tone)

        return

    def sound_timer(self) -> None:
        if self.sound_timer_register > 0:
            self.is_already_playing = False
            self.beep()
            self.sound_timer_register -= 1
        elif self.sound_timer_register <= 0:
            self.is_already_playing = True
            rl.stop_sound(self.tone)

    def delay_timer(self) -> None:
        if self.delay_timer_register > 0:
            self.delay_timer_register -= 1

    def render(self):
        for y in range(32):
            for x in range(64):
                if self.display[x][y] == 1:
                    rl.draw_rectangle(x * SCALE, y * SCALE, SCALE, SCALE, PIXEL_COLOR)
                else:
                    pass

    def load_rom(self, filename: str):
        assert filename.endswith(".ch8"), return_text_red("File does not end with .ch8")

        try:
            with open(filename, "rb") as file:
                data = file.read()
                n = 0
                for byte in data:
                    self.memory[self.PC + n] = byte
                    n += 1
        except FileNotFoundError:
            raise FileNotFoundError(return_text_red(f"File: {filename} does not exist"))

        except Exception as e:
            print(return_text_red(f"Exception: {e}"))

    def tick(self) -> None:
        # rate of 60hz
        self.delay_timer()
        self.sound_timer()
        self.exec_instruction()

    def vx_ld_vy(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        self.registers[x] = self.registers[y]

    def vx_or_vy(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        self.registers[x] = self.registers[x] | self.registers[y]

    def vx_and_vy(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        self.registers[x] = self.registers[x] & self.registers[y]

    def vx_xor_vy(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        self.registers[x] = self.registers[x] ^ self.registers[y]

    def vx_add_vy(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        res = self.registers[x] + self.registers[y]
        if res > 255:
            # VF is the last register
            self.registers[0xF] = 1
        else:
            self.registers[0xF] = 0

        self.registers[x] = res & 0xFF

    def vx_sub_vy(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        if self.registers[x] > self.registers[y]:
            self.registers[0xF] = 1
        else:
            self.registers[0xF] = 0

        res = self.registers[x] - self.registers[y]
        self.registers[x] = res

    def vx_shr_1(self, op: int) -> None:
        x = op >> 8
        Vx = self.registers[x]
        lsb_bit = Vx & 1
        if lsb_bit == 1:
            self.registers[0xF] = 1
        else:
            self.registers[0xF] = 0

        self.registers[x] //= 2

    def vx_subn_vy(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        Vx = self.registers[x]
        Vy = self.registers[y]

        if Vy > Vx:
            self.registers[0xF] = 1
        else:
            self.registers[0xF] = 0

        self.registers[x] = self.registers[y] - self.registers[x]

    def vx_shl_1(self, op: int) -> None:
        x = op >> 8
        Vx = self.registers[x]
        msb_bit = Vx & 128
        if msb_bit == 1:
            self.registers[0xF] = 1
        else:
            self.registers[0xF] = 0

        self.registers[x] *= 2

    def handle_0x8(self, op: int) -> None:
        last_op = op & 0xF
        match last_op:
            case 0x0:
                self.vx_ld_vy(op)
            case 0x1:
                self.vx_or_vy(op)
            case 0x2:
                self.vx_and_vy(op)
            case 0x3:
                self.vx_xor_vy(op)
            case 0x4:
                self.vx_add_vy(op)
            case 0x5:
                self.vx_sub_vy(op)
            case 0x6:
                self.vx_shr_1(op)
            case 0x7:
                self.vx_subn_vy(op)
            case 0xE:
                self.vx_shl_1(op)

    def clear_display(self) -> None:
        self.display = [[0] * 32 for _ in range(64)]

    def return_subroutine(self) -> None:
        self.PC = self.stack[self.SP]
        self.SP -= 1

    def jump_to(self, nnn: int) -> None:
        self.PC = nnn

    def call_subroutine(self, nnn: int) -> None:
        self.SP += 1
        self.stack[self.SP] = self.PC
        self.PC = nnn

    def skip_if_eq_kk(self, op: int) -> None:
        x = op >> 8
        kk = op & 0xFF
        Vx = self.registers[x]
        if Vx == kk:
            self.PC += 2

    def skip_if_not_eq_kk(self, op: int) -> None:
        x = op >> 8
        kk = op & 0xFF
        Vx = self.registers[x]
        if Vx != kk:
            self.PC += 2

    def skip_if_eq_vx(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        Vx = self.registers[x]
        Vy = self.registers[y]
        if Vx == Vy:
            self.PC += 2

    def ld_vx_to_kk(self, op: int) -> None:
        x = op >> 8
        kk = op & 0xFF
        self.registers[x] = kk

    def vx_add_kk(self, op: int) -> None:
        x = op >> 8
        kk = op & 0xFF
        self.registers[x] = self.registers[x] + kk

    def skip_if_not_eq_vx_vy(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        Vx = self.registers[x]
        Vy = self.registers[y]
        if Vx != Vy:
            self.PC += 2

    def set_i_to_nnn(self, nnn: int) -> None:
        self.I = nnn

    def jump_to_nnn_plus_v0(self, nnn: int) -> None:
        self.PC = nnn + self.registers[0x0]

    def set_vx_to_randbyte_and_kk(self, op: int) -> None:
        x = op >> 8
        kk = op & 0xFF
        randbyte = randint(0, 255)
        self.registers[x] = randbyte & kk

    def read_n_bytes(self, n) -> list[int]:
        sprites: list[int] = []
        for i in range(n):
            if self.I + n > len(self.memory):
                raise ValueError("Attempted to read beyond memory limits")
            sprites.append(self.memory[self.I + i])

        return sprites

    def display_sprite(self, op: int) -> None:
        x = op >> 8
        y = (op & 0xFF) >> 4
        n = op & 0xF
        sprite = self.read_n_bytes(n)
        # xPos = self.registers[x] % 64
        # yPos = self.registers[y] % 32
        xPos = self.registers[x]
        yPos = self.registers[y]
        # SKIBIDI
        self.registers[0xF] = 0

        for row in range(n):
            spriteByte = sprite[row]
            for col in range(8):
                spritePixel = (spriteByte >> (7 - col)) & 1
                screenX = (xPos + col) % 64
                screenY = (yPos + row) % 32

                if spritePixel == 1:
                    if self.display[screenX][screenY] == 1:
                        self.registers[0xF] = 1

                    self.display[screenX][screenY] ^= 1

    def handle_0x0(self, op: int) -> None:
        match op:
            case 0x0E0:
                self.clear_display()
            case 0x0EE:
                self.return_subroutine()
            case _:
                # Instruction is 0nnn but it is ignored by modern interpreters
                pass

    def skip_if_key_vx_pressed(self, op: int) -> None:
        x = op >> 8
        Vx = self.registers[x]
        key = self.convert_to_keyboard(Vx)
        if rl.is_key_down(key):
            self.PC += 2

    def skip_if_key_vx_not_pressed(self, op: int) -> None:
        x = op >> 8
        Vx = self.registers[x]
        key = self.convert_to_keyboard(Vx)
        if rl.is_key_up(key):
            self.PC += 2

    def convert_to_keyboard(self, key: int) -> int:
        match key:
            case 0x1:
                return rl.KeyboardKey.KEY_ONE.value
            case 0x2:
                return rl.KeyboardKey.KEY_TWO.value
            case 0x3:
                return rl.KeyboardKey.KEY_THREE.value
            case 0xC:
                return rl.KeyboardKey.KEY_FOUR.value
            case 0x4:
                return rl.KeyboardKey.KEY_Q.value
            case 0x5:
                return rl.KeyboardKey.KEY_W.value
            case 0x6:
                return rl.KeyboardKey.KEY_E.value
            case 0xD:
                return rl.KeyboardKey.KEY_R.value
            case 0x7:
                return rl.KeyboardKey.KEY_A.value
            case 0x8:
                return rl.KeyboardKey.KEY_S.value
            case 0x9:
                return rl.KeyboardKey.KEY_D.value
            case 0xE:
                return rl.KeyboardKey.KEY_F.value
            case 0xA:
                return rl.KeyboardKey.KEY_Z.value
            case 0x0:
                return rl.KeyboardKey.KEY_X.value
            case 0xB:
                return rl.KeyboardKey.KEY_C.value
            case 0xF:
                return rl.KeyboardKey.KEY_V.value
            case _:
                return rl.KeyboardKey.KEY_NULL.value

    def handle_0xE(self, op: int) -> None:
        last_op = op & 0xF
        match last_op:
            case 0xE:
                self.skip_if_key_vx_pressed(op)
            case 0x1:
                self.skip_if_key_vx_not_pressed(op)

    def vx_eq_dt(self, op: int) -> None:
        x = op >> 8
        self.registers[x] = self.delay_timer_register

    def wait_keypress_store_vx(self, op: int) -> None:
        x = op >> 8
        while True:
            for raylib_key, chip8_key in self.keyboard_map.items():
                if rl.is_key_pressed(raylib_key):
                    self.registers[x] = chip8_key
                    return

            self.delay_timer()
            self.sound_timer()

            rl.wait_time(1 / FPS)

    def dt_eq_vx(self, op: int) -> None:
        x = op >> 8
        self.delay_timer_register = self.registers[x]

    def st_eq_vx(self, op: int) -> None:
        x = op >> 8
        self.sound_timer_register = self.registers[x]

    def i_eq_i_plus_vx(self, op: int) -> None:
        x = op >> 8
        self.I = self.I + self.registers[x]

    def i_eq_location_sprite_vx(self, op: int) -> None:
        x = op >> 8
        Vx = self.registers[x]
        self.I = Vx * 5

    def bcd_repr_vx(self, op: int) -> None:
        x = op >> 8
        Vx = self.registers[x]
        self.memory[self.I] = Vx // 100
        self.memory[self.I + 1] = (Vx // 10) % 10
        self.memory[self.I + 2] = Vx % 10

    def copy_v0_to_vx(self, op: int) -> None:
        x = op >> 8
        for i in range(x):
            self.memory[self.I + i] = self.registers[i]

    def read_v0_to_vx(self, op: int) -> None:
        x = op >> 8
        for i in range(x):
            self.registers[i] = self.memory[self.I + i]

    def handle_0xF(self, op: int) -> None:
        last_op = op & 0xFF
        match last_op:
            case 0x07:
                self.vx_eq_dt(op)
            case 0x0A:
                self.wait_keypress_store_vx(op)
            case 0x15:
                self.dt_eq_vx(op)
            case 0x18:
                self.st_eq_vx(op)
            case 0x1E:
                self.i_eq_i_plus_vx(op)
            case 0x29:
                self.i_eq_location_sprite_vx(op)
            case 0x33:
                self.bcd_repr_vx(op)
            case 0x55:
                self.copy_v0_to_vx(op)
            case 0x65:
                self.read_v0_to_vx(op)

    def exec_instruction(self) -> None:
        instruction: int = (self.memory[self.PC] << 8) | (self.memory[self.PC + 1])
        opcode: int = instruction >> 12
        operands: int = instruction & 0xFFF
        self.PC += 2
        print(
            f"INSTRUCTION: {hex(instruction)}, OPCODE: {hex(opcode)}, OPERANDS: {hex(operands)}"
        )

        match (opcode, operands):
            case (0x0, op):
                self.handle_0x0(op)

            case (0x1, op):
                self.jump_to(op)

            case (0x2, op):
                self.call_subroutine(op)

            case (0x3, op):
                self.skip_if_eq_kk(op)

            case (0x4, op):
                self.skip_if_not_eq_kk(op)

            case (0x5, op):
                self.skip_if_eq_vx(op)

            case (0x6, op):
                self.ld_vx_to_kk(op)

            case (0x7, op):
                self.vx_add_kk(op)

            case (0x8, op):
                self.handle_0x8(op)

            case (0x9, op):
                self.skip_if_not_eq_vx_vy(op)

            case (0xA, op):
                self.set_i_to_nnn(op)

            case (0xB, op):
                self.jump_to_nnn_plus_v0(op)

            case (0xC, op):
                self.set_vx_to_randbyte_and_kk(op)

            case (0xD, op):
                self.display_sprite(op)

            case (0xE, op):
                self.handle_0xE(op)

            case (0xF, op):
                self.handle_0xF(op)

    def reset(self) -> None:
        # Before you reset the chip8 state
        # make sure you do init_emu() at initialization and close_emu() at termination
        self.memory: list[int] = [0] * 4096
        self.registers: list[int] = [0] * 16
        self.I: int = 0
        self.delay_timer_register: int = 0
        self.sound_timer_register: int = 0
        self.PC: int = 0x200
        self.SP: int = 0
        self.stack: list[int] = [0] * 16
        self.display: list[list[int]] = [[0] * 32 for _ in range(64)]
        self.fonts: list[int] = font_codes
        self.tone = None
        self.is_already_playing = False


def return_text_red(txt):
    return "\033[91m" + txt + "\033[0m"


# Config Variables
SCALE = 15
WIDTH = 64 * SCALE
HEIGHT = 32 * SCALE
TITLE = "Chip8 Emu"
FPS = 400
BG_COLOR = rl.BLACK
PIXEL_COLOR = rl.GREEN
THRESHOLD = 1 / FPS


def get_rom():
    rom: str = sys.argv[1]
    if rom.strip() == "":
        print(return_text_red("No file was given"))
        sys.exit(1)
    return rom


def main(rom):
    # Init
    rl.init_window(WIDTH, HEIGHT, TITLE)
    rl.set_window_position(20, 60)
    rl.set_target_fps(FPS)

    # Chip8 Init
    c = Chip8()

    # FIXME: Fix keypad.ch8 test not rendering choices prperly?
    # I don't know why
    c.load_rom(rom)
    c.init_emu()

    accumulator = 0

    while not rl.window_should_close():
        rl.begin_drawing()
        accumulator += rl.get_frame_time()

        while accumulator >= THRESHOLD:
            c.tick()
            accumulator -= THRESHOLD

        # if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE):
        #     c.tick()

        c.render()
        rl.clear_background(BG_COLOR)
        rl.end_drawing()

    c.close_emu()
    rl.close_window()


if __name__ == "__main__":
    main(get_rom())

import { afterEach, describe, expect, it, vi } from "vitest";
import { downloadText, safeDownloadFilename } from "./export";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("downloadText", () => {
  it("defers object URL revocation until after the download click", () => {
    vi.useFakeTimers();
    const createObjectURL = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:capswriter-download");
    const revokeObjectURL = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    downloadText("transcript.txt", "hello");

    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(revokeObjectURL).not.toHaveBeenCalled();
    expect(document.querySelector('a[download="transcript.txt"]')).toBeNull();

    vi.runOnlyPendingTimers();

    expect(revokeObjectURL).toHaveBeenCalledWith("blob:capswriter-download");
  });

  it("sanitizes the download attribute", () => {
    vi.useFakeTimers();
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:capswriter-download");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function clickAnchor(this: HTMLAnchorElement) {
        expect(this.download).toBe("evil-name.txt");
      });

    downloadText("../evil\nname.txt", "hello");

    expect(click).toHaveBeenCalledOnce();
  });
});

describe("safeDownloadFilename", () => {
  it("replaces path separators and reserved filename characters", () => {
    expect(safeDownloadFilename(' ..\\meeting:Q1/notes?.txt ')).toBe(
      "meeting-Q1-notes-.txt",
    );
  });

  it("uses a fallback when the filename becomes empty", () => {
    expect(safeDownloadFilename(" ../\n\t ")).toBe("capswriter-download.txt");
  });

  it("prefixes Windows reserved device filenames", () => {
    expect(safeDownloadFilename("CON.txt")).toBe("capswriter-CON.txt");
    expect(safeDownloadFilename("nul")).toBe("capswriter-nul");
  });
});

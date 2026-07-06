import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  it("renders the primary workbench regions", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "Web Console" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "連線" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "音訊" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "轉錄" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "TTS" })).toBeTruthy();
  });
});

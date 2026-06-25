(function () {
  const root = document.documentElement;
  const storedTheme = localStorage.getItem("qayd-docs-theme");
  if (storedTheme === "dark" || storedTheme === "light") {
    root.dataset.theme = storedTheme;
  }

  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      localStorage.setItem("qayd-docs-theme", next);
    });
  });

  const copyText = async (text, button) => {
    try {
      await navigator.clipboard.writeText(text.trim());
      button.classList.add("copied");
      const label = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = label;
        button.classList.remove("copied");
      }, 1200);
    } catch {
      button.textContent = "Copy failed";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1200);
    }
  };

  document.querySelectorAll("[data-copy], [data-copy-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const literal = button.getAttribute("data-copy");
      const target = button.getAttribute("data-copy-target");
      const text = literal || (target ? document.querySelector(target)?.textContent : "");
      if (text) {
        copyText(text, button);
      }
    });
  });

  // Tabbed code card: clicking a tab shows its panel.
  document.querySelectorAll("[data-tabs]").forEach((card) => {
    const tabs = card.querySelectorAll(".code-tab");
    const panels = card.querySelectorAll("[data-panel]");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const name = tab.getAttribute("data-tab");
        tabs.forEach((t) => {
          const active = t === tab;
          t.classList.toggle("is-active", active);
          t.setAttribute("aria-selected", active ? "true" : "false");
        });
        panels.forEach((panel) => {
          panel.hidden = panel.getAttribute("data-panel") !== name;
        });
      });
    });
  });
})();

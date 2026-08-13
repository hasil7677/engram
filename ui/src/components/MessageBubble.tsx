import type { ChatMessage } from "../types";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const isEmpty = message.streaming && message.text.length === 0;

  return (
    <div className={`message-row ${message.role}`}>
      <div className={`message-bubble ${message.error ? "error" : ""}`}>
        {isEmpty ? (
          <span className="thinking-dots">
            <span />
            <span />
            <span />
          </span>
        ) : (
          <>
            {message.text}
            {message.streaming && <span className="cursor" />}
          </>
        )}
      </div>
    </div>
  );
}
